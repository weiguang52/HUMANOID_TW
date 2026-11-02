"""Declared offline arithmetic contract; original artifacts keep their old behavior."""
import copy,hashlib
import torch
CONTRACT='offline_internal_float64_preserve_api_dtype_v1'

def state_digest(state):
 digest=hashlib.sha256()
 for key,value in sorted(state.items()):
  tensor=value.detach().cpu().contiguous()
  digest.update(key.encode());digest.update(str(tensor.dtype).encode());digest.update(str(tuple(tensor.shape)).encode());digest.update(tensor.numpy().tobytes())
 return digest.hexdigest()

def metadata(source_sha,weights_sha):
 return dict(contract=CONTRACT,internal_dtype='float64',api_dtype='input_state_dtype',stored_weights_dtype='float32_parameters_original_buffers',source_checkpoint_sha256=source_sha,stored_world_model_sha256=weights_sha,scope='offline_only')

def validate(artifact):
 meta=artifact.get('precision_contract')
 if meta is None:return False
 if artifact.get('kind')!='offline_yaw_canonical_dynamics_not_policy_checkpoint':raise ValueError('Precision contract requires explicit offline yaw artifact')
 for key in ['source_checkpoint_sha256','stored_world_model_sha256']:
  value=meta.get(key);assert isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
 if meta!=metadata(meta['source_checkpoint_sha256'],meta['stored_world_model_sha256']):raise ValueError('Unsupported precision contract')
 assert artifact.get('offline_only') is True and artifact.get('resumable_training') is False and artifact.get('policy_promotion_approved') is False
 assert 'optimizer' not in artifact
 assert artifact['calibration_plan'].get('precision_contract')==meta
 assert artifact['coordinate_contract']['network_dtype']=='float64'
 counts={'state_norm.count','action_norm.count','context_norm.count','goal_norm.count'}
 assert all(not v.is_floating_point() or v.dtype==(torch.float64 if k in counts else torch.float32) for k,v in artifact['world_model'].items())
 assert state_digest(artifact['world_model'])==meta['stored_world_model_sha256']
 return True

def prepare_precision_model(model,artifact):
 if not validate(artifact):return model
 return copy.deepcopy(model).double().eval()

class DeclaredFloat64Predictor(torch.nn.Module):
 def __init__(self,predictor,meta):
  super().__init__()
  if not all(v.dtype==torch.float64 for v in predictor.parameters()):raise ValueError('Use complete precision prediction loader')
  self.predictor=predictor.eval();self.precision_contract=dict(meta)
 def forward(self,state,action,context):
  if state.dtype not in [torch.float32,torch.float64]:raise ValueError('Declared arithmetic API accepts float32/float64 state')
  with torch.autocast(device_type=state.device.type,enabled=False):out=self.predictor(state.double(),action.double(),context.double())
  return out.to(state.dtype)

def apply_precision_contract(predictor,artifact):
 if not validate(artifact):return predictor
 return DeclaredFloat64Predictor(predictor,artifact['precision_contract'])
