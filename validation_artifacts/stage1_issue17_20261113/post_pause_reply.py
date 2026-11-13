import json,sys,urllib.request
from pathlib import Path
p=Path('/root/gpufree-data/tw_stage1/issue17_20261113');token=sys.stdin.readline().strip();assert token
api='https://api.github.com/repos/weiguang52/HUMANOID_TW'
def req(method,path,data=None):
 r=urllib.request.Request(api+path,method=method,data=None if data is None else json.dumps(data).encode(),headers={'Authorization':'Bearer '+token,'User-Agent':'TW-issue17-preflight','Content-Type':'application/json','Accept':'application/vnd.github+json'})
 with urllib.request.urlopen(r,timeout=30) as response:return json.load(response)
body='''Paused at the user\'s request before formal training. No issue17 training has started, and no reward/production configuration change has been made.

Preflight findings:
- The training-only spatial-reflection + half-cycle transform is implemented separately from issue16 pure-spatial evaluation. Four tests pass: actor101/critic134/action30 involution, correct clock inversion, critic vector/joint blocks, and TensorDict augmentation ordering/optional inputs/RNG neutrality.
- The required exact reward-invariance test does NOT pass with the unchanged current floating-point clock. Testing the actual AST-extracted feet_gait and feet_swing_height functions on4000 synthetic CPU states (integer steps0..999 repeated, left/right contacts or foot heights swapped, episode step+10) gives196 contact-reward mismatches and162 swing-height mismatches; maximum reward difference1.0 in each test. These are synthetic preflight counts, not rollout failure rates or training outcomes.
- Every stance-mask mismatch is near a phase threshold or modulo boundary (maximum boundary distance1.8477439880371094e-6). For example phase0.5500000119 versus0.5499999523 falls on opposite sides of the strict0.55 threshold. A tiny phase error can therefore cause a finite discrete reward change.
- Both actual reward functions are exactly invariant on the3200 tested states away from boundaries. An integer20-step clock oracle with stance threshold11/20 is exactly invariant under a10-step shift. This supports a numerical-boundary diagnosis for these tests, not a claim that the unchanged implementation passes the strict requirement or that all other reward/MDP invariants have been proved.

The issue currently requires both unchanged rewards/reused control6000 and elementwise reward equality before training. Changing the clock could alter rewards and compromise the existing control comparison; excluding boundary cases would weaken the stated exact test. Neither change has been silently applied, and the failed test is not marked passed.

Evidence remains on the remote data disk at /root/gpufree-data/tw_stage1/issue17_20261113/: protocol_plan.md, transform_unit_tests.log, reward_phase_initial_check.json, phase_boundary_diagnosis.log and the reproducer scripts. Implementation/tests are uncommitted; formal budgets remain unused. Awaiting an explicit decision on whether to retain the original clock with disclosed boundary exceptions, authorize a corrected clock with a new matched control, or keep the strict prerequisite and stop before training. The long-term goal is paused; no extra training, checkpoint search or phase-scheme change will proceed while paused.
'''
comments=req('GET','/issues/17/comments?per_page=100');same=[c for c in comments if c['body']==body];assert len(same)<=1
c=same[0] if same else req('POST','/issues/17/comments',{'body':body});r=req('GET','/issues/comments/'+str(c['id']));assert r['body']==body
(p/'pause_issue_reply_receipt.json').write_text(json.dumps(dict(url=c['html_url'],exact_readback=True,body=body),indent=2)+'\n');print('PAUSE_REPLY_VERIFIED',c['html_url'])
