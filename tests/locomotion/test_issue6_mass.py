"""Exercise installed Isaac Lab mass sampling with a fake PhysX tensor view."""
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
import unittest
from types import SimpleNamespace as NS
import torch
from isaaclab.envs.mdp.events import randomize_rigid_body_mass
from unitree_rl_lab.tasks.locomotion.robots.tw.mass_randomization import configure_mass_randomization

class View:
    def __init__(self,x):self.value=x.clone()
    def get_masses(self):return self.value.clone()
    def set_masses(self,x,ids):self.value[ids]=x[ids]

class MassBoundsTests(unittest.TestCase):
    def fixture(self,n=4096):
        nominal=torch.tensor([.07990783598,.001,2.]).repeat(n,1)
        view=View(nominal)
        asset=NS(num_bodies=3,data=NS(default_mass=nominal),root_physx_view=view)
        state=NS(asset=asset,asset_cfg=NS(body_ids=[0]))
        return nominal,view,state,NS(scene=NS(num_envs=n))
    def configured_params(self):
        friction=NS(params={"static_friction_range":(.4,1.4),"dynamic_friction_range":(.3,1.2)})
        cfg=NS(events=NS(add_base_mass=NS(params={"operation":"add","mass_distribution_params":(-.08,.15)}),physics_material=friction))
        configure_mass_randomization(cfg,"bounded")
        self.assertEqual(friction.params,{"static_friction_range":(.4,1.4),"dynamic_friction_range":(.3,1.2)})
        return cfg.events.add_base_mass.params
    def test_uniform_bound_real_sampler_and_untargeted_bodies(self):
        params=self.configured_params()
        for seed in [1,2,3]:
            torch.manual_seed(seed);nominal,view,state,env=self.fixture()
            randomize_rigid_body_mass.__call__(state,env,None,asset_cfg=None,recompute_inertia=False,**params)
            self.assertTrue((view.value[:,0]>=nominal[:,0]*.5).all())
            self.assertTrue((view.value[:,0]<=nominal[:,0]*1.5).all())
            self.assertTrue(torch.equal(view.value[:,1:],nominal[:,1:]))
            self.assertFalse((view.value<=1.e-6).any())
    def test_lower_endpoint_and_partial_reset_no_accumulation(self):
        nominal,view,state,env=self.fixture();ids=torch.tensor([0,2])
        for _ in range(2):
            randomize_rigid_body_mass.__call__(state,env,ids,asset_cfg=None,mass_distribution_params=(.5,.5),operation="scale",recompute_inertia=False)
            self.assertTrue(torch.equal(view.value[ids,0],nominal[ids,0]*.5))
            self.assertTrue(torch.equal(view.value[1],nominal[1]))
    def test_legacy_negative_draw_hits_clamp(self):
        _,view,state,env=self.fixture()
        randomize_rigid_body_mass.__call__(state,env,None,asset_cfg=None,mass_distribution_params=(-.08,-.08),operation="add",recompute_inertia=False)
        self.assertTrue((view.value[:,0]==1.e-6).all())
    def test_opt_in_preserves_legacy_and_disabled_term(self):
        cfg=NS(events=NS(add_base_mass=NS(params={"operation":"add","mass_distribution_params":(-.08,.15)})))
        configure_mass_randomization(cfg,"legacy")
        self.assertEqual(cfg.events.add_base_mass.params,{"operation":"add","mass_distribution_params":(-.08,.15)})
        cfg.events.add_base_mass=None
        configure_mass_randomization(cfg,"bounded");self.assertIsNone(cfg.events.add_base_mass)

suite=unittest.defaultTestLoader.loadTestsFromTestCase(MassBoundsTests)
result=unittest.TextTestRunner(verbosity=2).run(suite)
app.close()
raise SystemExit(0 if result.wasSuccessful() else 1)
