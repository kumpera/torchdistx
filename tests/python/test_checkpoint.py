from typing import Optional

import torch
import torch.distributed as dist
import torch.nn

from torchdistx.checkpoint.state_dict_loader import  validate_metadata
from torchdistx.checkpoint.state_dict_saver import _prepare
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharded_tensor import (
    state_dict_hook,
    ShardedTensor,
)
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)


class TestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sharded: ShardedTensor = sharded_tensor.zeros(self.spec(), 4, 4)
        self.regular = torch.nn.Parameter(torch.ones(4, 4))
        self.extra_sharded: Optional[ShardedTensor] = None
        self.extra_param: Optional[torch.nn.Parameter] = None
        self._register_state_dict_hook(state_dict_hook)

    def spec(self) -> ChunkShardingSpec:
        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        return ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        )


class TestCheckpointing(ShardedTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    # pyre-fixme [56]: Pyre was not able to infer the type of the decorator `torch.testing._internal.common_distributed.requires_nccl()`
    @requires_nccl()
    def test_validate_metadata(self) -> None:
        module = TestModule()
        # compute the default saved metadata (must pass always_add_tensors or we'll get incomplete MD)
        metadata, _, _, _ = _prepare(module.state_dict(), always_add_tensors=True)
        self.assertTrue(
            "regular" in metadata.state_dict_metadata,
            f"keys: {metadata.state_dict_metadata.keys()}",
        )

        module = TestModule()
        self.assertIsNone(validate_metadata(module.state_dict(), metadata))

        module = TestModule()
        module.extra_param = torch.nn.Parameter(torch.zeros(2, 2))
        res = validate_metadata(module.state_dict(), metadata)
        self.assertIsNotNone(res)
        self.assertTrue("Could not find Tensor metadata" in res[0])
        self.assertTrue("extra_param" in res[0])

        module = TestModule()
        module.regular = torch.nn.Parameter(torch.zeros(2, 4))

        res = validate_metadata(module.state_dict(), metadata)
        self.assertIsNotNone(res)
        self.assertTrue("Incompatible tensor size" in res[0])
        self.assertTrue("regular" in res[0])

        module = TestModule()
        module.extra_sharded = sharded_tensor.zeros(module.spec(), 4, 2)
        res = validate_metadata(module.state_dict(), metadata)
        self.assertIsNotNone(res)
        self.assertTrue("Could not find ShardedTensor metadata" in res[0])
        self.assertTrue("extra_sharded" in res[0])

        module = TestModule()
        module.sharded = sharded_tensor.zeros(module.spec(), 4, 2)
        res = validate_metadata(module.state_dict(), metadata)
        self.assertIsNotNone(res)
        self.assertTrue("Incompatible ShardedTensor size" in res[0])
        self.assertTrue("sharded" in res[0])

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    # pyre-fixme [56]: Pyre was not able to infer the type of the decorator `torch.testing._internal.common_distributed.requires_nccl()`
    @requires_nccl()
    def test_metadata_is_different_across_ranks(self) -> None:
        module = TestModule()
        # compute the default saved metadata (must pass always_add_tensors or we'll get incomplete MD)
        metadata, _, _, _ = _prepare(module.state_dict(), always_add_tensors=False)

        # _prepare skips tensors when rank > 0
        if dist.get_rank() == 0:
            self.assertTrue(
                "regular" in metadata.state_dict_metadata,
                f"keys: {metadata.state_dict_metadata.keys()}",
            )
        else:
            self.assertTrue(
                "regular" not in metadata.state_dict_metadata,
                f"keys: {metadata.state_dict_metadata.keys()}",
            )
