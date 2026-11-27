from hcu_sglang.utils.detection import is_hcu


def around_pp_sync_ready_failed(original, self, ready_req_idxs, failed_req_idxs):
    if self.pp_rank == 0 and is_hcu():
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        if parallel.attn_dp_enabled and parallel.attn_cp_size > 1:
            import torch.distributed as dist

            group = parallel.attn_cp_group
            gathered = [None] * group.world_size
            dist.all_gather_object(
                gathered, (ready_req_idxs, failed_req_idxs), group=group.cpu_group
            )
            ready_req_idxs = set.intersection(*(ready for ready, _ in gathered))
            failed_req_idxs = set.union(*(failed for _, failed in gathered))

    return original(self, ready_req_idxs, failed_req_idxs)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.constrained.grammar_manager.GrammarManager._pp_sync_ready_failed",
        around_pp_sync_ready_failed,
        HookType.AROUND,
    )
