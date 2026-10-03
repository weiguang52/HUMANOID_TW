import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/practice9'))
from prepare_tw75_training import eligible
class AdmissionTests(unittest.TestCase):
    def record(self,reasons,clipped=0):
        return dict(quality_pass=False,quality={'reasons':reasons},clipped_fraction=clipped)
    def test_only_reviewed_near_limit_warnings_are_advisory(self):
        self.assertTrue(eligible(self.record(['joint_near_limit_fraction','joint_near_limit_run'])))
        for reason in ['joint_speed_max','required_time_scale_exceeds_cap','root_xy_speed','unknown']:
            with self.subTest(reason=reason):
                self.assertFalse(eligible(self.record(['joint_near_limit_run',reason])))
    def test_clipping_and_missing_evidence_are_not_accepted(self):
        self.assertFalse(eligible(self.record(['joint_near_limit_run'],.001)))
        self.assertFalse(eligible(self.record([])))
        self.assertFalse(eligible({}))
    def test_strict_pass_remains_eligible(self):
        self.assertTrue(eligible(dict(quality_pass=True,clipped_fraction=0)))
