"""Strict original seed42 loader, observed hook counts and ordinary frozen caches."""
import hashlib
import torch
import numpy as np
from depth_common import require, module, state_hash
from fusion_adapter import FrozenFusionModel


def tensor_receipt(tensor):
    value=tensor.detach().cpu().contiguous()
    require(bool(torch.isfinite(value).all()),'Nonfinite retained tensor receipt')
    return dict(shape=list(value.shape),dtype=str(value.dtype),
                sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                l2=float(value.double().square().sum().sqrt()),
                absolute_max=float(value.abs().max()))


def bitwise_equal(left,right):
    return left.shape==right.shape and left.dtype==right.dtype and torch.equal(left.view(torch.int32),right.view(torch.int32))


class FrozenRuntime:
    def __init__(self,plan,adapter,arm):
        loader=module(plan['source_code']['model_contract_v2.py'],'phase20_strict_seed42_loader')
        runtime=loader.load_runtime()
        model,metadata=loader.build_model(plan['checkpoint_path'],plan['checkpoint'],runtime)
        model.requires_grad_(False);model.eval()
        require(sum(p.numel() for p in model.parameters())==34546498,'Original capacity changed')
        self.model=model;self.metadata=metadata;self.arm=arm;self.adapter=adapter
        self.wrapper=FrozenFusionModel(model,adapter,arm)
        self.before=self.snapshot()
        require(self.before['state_sha256']==plan['backbone_state_sha256'],'Original semantic state differs')
        self.counts=dict(original_full_model=0,original_depth_fusion=0,original_2d_network=0,
                         adapter_forwards=0,backwards=0,optimizer_steps=0,decoder_input_backwards=0)
        self.handles=[];self.captured=None
        for name,part in [('original_full_model',model),('original_depth_fusion',model.depth_fusion),
                          ('original_2d_network',model.network),('adapter_forwards',adapter)]:
            def hook(module,inputs,output,name=name):
                self.counts[name]+=1
            self.handles.append(part.register_forward_hook(hook))

    def snapshot(self):
        return dict(state_sha256=state_hash(self.model.state_dict()),
                    modes={name:part.training for name,part in self.model.named_modules()},
                    named_buffers=list(dict(self.model.named_buffers())),
                    trainable_parameters=[name for name,p in self.model.named_parameters() if p.requires_grad],
                    gradient_parameters=[name for name,p in self.model.named_parameters() if p.grad is not None])

    def check(self):
        require(all(not part.training for part in self.model.modules())
                and all(not p.requires_grad and p.grad is None for p in self.model.parameters()),
                'Original modes, trainability or gradients changed')

    def original_and_fused(self,image,teacher):
        self.check();captured=[]
        def capture(module,inputs,output):captured.append(output.detach())
        handle=self.model.depth_fusion.register_forward_hook(capture)
        try:
            with torch.no_grad():original=self.model(image)['ink']
        finally:handle.remove()
        require(len(captured)==1 and bitwise_equal(original,teacher),
                'Actual original seed42 output differs bitwise from exact-input cached teacher')
        fused=captured[0]
        require(fused.shape==(image.shape[0],32,*image.shape[3:]) and not fused.requires_grad
                and not torch.is_inference(fused),'Invalid ordinary detached fusion cache')
        return original,fused

    def prepare_training(self,batches,directory):
        receipts=[]
        arrays={name:np.lib.format.open_memmap(directory/(name+'.npy'),mode='w+',dtype=np.float32,shape=(32,2,1,128,128))
                for name in ['initial-original-logits','initial-zero-adapter-logits']}
        for batch in batches:
            image=batch['image'];original,fused=self.original_and_fused(image,batch['teacher'])
            with torch.no_grad():logits,residual,adapted=self.wrapper.forward_from_fused(image,fused)
            require(bitwise_equal(logits,original) and bitwise_equal(adapted,fused)
                    and not bool(residual.any()),'All-input zero-adapter identity failed')
            arrays['initial-original-logits'][batch['step']-1]=original.numpy()
            arrays['initial-zero-adapter-logits'][batch['step']-1]=logits.numpy()
            batch['fused']=fused
            receipt=dict(step=batch['step'],input_sha256=batch['input_sha256'],
                         original_state_sha256=self.before['state_sha256'],
                         original_modes_all_eval=True,ordinary_detached_tensor=True,
                         fused=tensor_receipt(fused),teacher=tensor_receipt(batch['teacher']),
                         original_logits=tensor_receipt(original),zero_adapter_logits=tensor_receipt(logits),
                         original_teacher_bitwise_equal=True,zero_adapter_original_bitwise_equal=True,
                         zero_residual=True,zero_adapted_features_bitwise_equal=True)
            batch['fused_sha256']=receipt['fused']['sha256'];receipts.append(receipt)
        for array in arrays.values():array.flush()
        arrays.clear()
        require(self.snapshot()==self.before,'Original state changed during initialization')
        return receipts

    def forward(self,batch):
        self.check()
        require(hashlib.sha256(batch['fused'].numpy().tobytes()).hexdigest()==batch['fused_sha256'],
                'Frozen feature cache changed')
        z,residual,adapted=self.wrapper.forward_from_fused(batch['image'],batch['fused'])
        if torch.is_grad_enabled():
            require(adapted.requires_grad,'Decoder input lost adapter gradient path')
            def record_gradient(gradient):
                self.counts['decoder_input_backwards']+=1
                self.last_decoder_input_gradient=tensor_receipt(gradient)
                require(self.last_decoder_input_gradient['l2']>0,'No gradient traversed frozen decoder')
                return gradient
            adapted.register_hook(record_gradient)
        return z,residual,adapted

    def finish(self,expected):
        self.check();after=self.snapshot()
        require(after==self.before,'Original parameters, buffers, modes or gradients changed')
        require(self.counts==expected,'Observed actual hook counts differ from fixed schedule')
        for handle in self.handles:handle.remove()
        return dict(state_sha256=self.before['state_sha256'],before=self.before,after=after,
                    unchanged=True,checkpoint_loads=1,checkpoint_metadata=self.metadata,
                    actual_calls=dict(self.counts),updates=0,parameter_gradients_absent=True,
                    count_semantics='Full-model hook events overlap their depth_fusion and 2D-network submodule events. Backwards traverse the frozen 2D network to adapter inputs; no original parameter gradients.')
