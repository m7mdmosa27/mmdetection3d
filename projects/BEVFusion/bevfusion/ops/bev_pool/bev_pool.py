import torch

from . import bev_pool_ext


class QuickCumsum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept, ) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class QuickCumsumCuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        # #region agent log
        try:
            with open('/home/ubuntu/mmdetection3d/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"id":"log_quickcumsum_entry","timestamp":int(__import__('time').time()*1000),"location":"bev_pool.py:39","message":"QuickCumsumCuda.forward entry","data":{"x_shape":list(x.shape) if x.numel() > 0 else [0],"x_numel":x.numel(),"geom_feats_shape":list(geom_feats.shape) if geom_feats.numel() > 0 else [0],"ranks_shape":list(ranks.shape) if ranks.numel() > 0 else [0],"B":int(B),"D":int(D),"H":int(H),"W":int(W),"hypothesisId":"A"},"sessionId":"debug-session","runId":"initial"}) + '\n')
        except: pass
        # #endregion agent log
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        # #region agent log
        try:
            with open('/home/ubuntu/mmdetection3d/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"id":"log_kept_info","timestamp":int(__import__('time').time()*1000),"location":"bev_pool.py:42","message":"kept tensor info","data":{"kept_shape":list(kept.shape) if kept.numel() > 0 else [0],"kept_sum":int(kept.sum().item()) if kept.numel() > 0 else 0,"kept_all_same":bool((kept[1:] == kept[:-1]).all().item()) if kept.numel() > 1 else True,"hypothesisId":"B"},"sessionId":"debug-session","runId":"initial"}) + '\n')
        except: pass
        # #endregion agent log
        interval_starts = torch.where(kept)[0].int()
        # #region agent log
        try:
            with open('/home/ubuntu/mmdetection3d/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"id":"log_interval_starts","timestamp":int(__import__('time').time()*1000),"location":"bev_pool.py:43","message":"interval_starts created","data":{"interval_starts_shape":list(interval_starts.shape) if interval_starts.numel() > 0 else [0],"interval_starts_numel":interval_starts.numel(),"interval_starts_empty":interval_starts.numel() == 0,"hypothesisId":"A"},"sessionId":"debug-session","runId":"initial"}) + '\n')
        except: pass
        # #endregion agent log
        interval_lengths = torch.zeros_like(interval_starts)
        # #region agent log
        try:
            with open('/home/ubuntu/mmdetection3d/.cursor/debug.log', 'a') as f:
                f.write(__import__('json').dumps({"id":"log_before_assign","timestamp":int(__import__('time').time()*1000),"location":"bev_pool.py:45","message":"Before interval_lengths assignment","data":{"interval_lengths_shape":list(interval_lengths.shape) if interval_lengths.numel() > 0 else [0],"interval_lengths_numel":interval_lengths.numel(),"x_shape_0":int(x.shape[0]) if x.numel() > 0 else 0,"interval_starts_last":int(interval_starts[-1].item()) if interval_starts.numel() > 0 else -1,"hypothesisId":"A"},"sessionId":"debug-session","runId":"initial"}) + '\n')
        except: pass
        # #endregion agent log
        
        # Handle empty input case: when all points are filtered out, interval_lengths will be empty
        if interval_lengths.numel() == 0:
            # #region agent log
            try:
                with open('/home/ubuntu/mmdetection3d/.cursor/debug.log', 'a') as f:
                    f.write(__import__('json').dumps({"id":"log_empty_handled","timestamp":int(__import__('time').time()*1000),"location":"bev_pool.py:69","message":"Handling empty input case","data":{"x_shape":list(x.shape) if x.numel() > 0 else [0],"x_numel":x.numel(),"B":int(B),"D":int(D),"H":int(H),"W":int(W),"hypothesisId":"A"},"sessionId":"debug-session","runId":"post-fix"}) + '\n')
            except: pass
            # #endregion agent log
            # Return zero tensor with correct output shape: (B, D, H, W, C)
            # When x is empty after filtering, it should still have shape (0, C) where C is preserved
            # If x.shape is 1D [0], we need to infer C from context or use a default
            if len(x.shape) >= 2:
                C = x.shape[1]
            else:
                # Fallback: infer from output gradient shape if available, or use default
                # This should rarely happen if filtering preserves tensor dimensions
                C = 1
            out = torch.zeros((B, D, H, W, C), device=x.device, dtype=x.dtype)
        else:
            interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
            interval_lengths[-1] = x.shape[0] - interval_starts[-1]
            geom_feats = geom_feats.int()

            out = bev_pool_ext.bev_pool_forward(
                x,
                geom_feats,
                interval_lengths,
                interval_starts,
                B,
                D,
                H,
                W,
            )

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        x_grad = bev_pool_ext.bev_pool_backward(
            out_grad,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        return x_grad, None, None, None, None, None, None


def bev_pool(feats, coords, B, D, H, W):
    assert feats.shape[0] == coords.shape[0]
    # #region agent log
    try:
        with open('/home/ubuntu/mmdetection3d/.cursor/debug.log', 'a') as f:
            f.write(__import__('json').dumps({"id":"log_bev_pool_entry","timestamp":int(__import__('time').time()*1000),"location":"bev_pool.py:83","message":"bev_pool function entry","data":{"feats_shape":list(feats.shape) if feats.numel() > 0 else [0],"feats_numel":feats.numel(),"coords_shape":list(coords.shape) if coords.numel() > 0 else [0],"B":int(B),"D":int(D),"H":int(H),"W":int(W),"feats_empty":feats.numel() == 0,"hypothesisId":"A"},"sessionId":"debug-session","runId":"initial"}) + '\n')
    except: pass
    # #endregion agent log

    # Handle empty input early - if all points were filtered out
    if feats.numel() == 0:
        # Preserve feature dimension C from feats.shape[1] if available
        C = feats.shape[1] if len(feats.shape) >= 2 else 1
        # Return zero tensor with correct output shape: (B, D, H, W, C) -> permuted to (B, C, D, H, W)
        x = torch.zeros((B, D, H, W, C), device=feats.device, dtype=feats.dtype)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    ranks = (
        coords[:, 0] * (W * D * B) + coords[:, 1] * (D * B) +
        coords[:, 2] * B + coords[:, 3])
    indices = ranks.argsort()
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]
    # #region agent log
    try:
        with open('/home/ubuntu/mmdetection3d/.cursor/debug.log', 'a') as f:
            ranks_unique = int(ranks.unique().numel()) if ranks.numel() > 0 else 0
            ranks_all_same = bool((ranks == ranks[0]).all().item()) if ranks.numel() > 0 else True
            f.write(__import__('json').dumps({"id":"log_ranks_info","timestamp":int(__import__('time').time()*1000),"location":"bev_pool.py:90","message":"ranks after sorting","data":{"ranks_shape":list(ranks.shape) if ranks.numel() > 0 else [0],"ranks_unique_count":ranks_unique,"ranks_all_same":ranks_all_same,"hypothesisId":"B"},"sessionId":"debug-session","runId":"initial"}) + '\n')
    except: pass
    # #endregion agent log

    x = QuickCumsumCuda.apply(feats, coords, ranks, B, D, H, W)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x
