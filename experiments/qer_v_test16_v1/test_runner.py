import unittest
from runner import summarize, check_score, MODULES, METHODS


class TestEvaluation(unittest.TestCase):
    def records(self):
        return [dict(module=n,method=m,window=w,scores=dict(tokens=2047,KL=(w+1.) if m=='None' else 1.))
                for n in MODULES for m in METHODS for w in range(16)]

    def test_recovery_uses_aggregate_not_average_window_ratios(self):
        rows=summarize(self.records())
        self.assertEqual(len(rows),24)
        for r in rows:
            self.assertAlmostEqual(r['recovery_percent'],0. if r['method']=='None' else 100*(1-1/8.5))

    def test_missing_and_duplicate_windows_rejected(self):
        rows=self.records()
        with self.assertRaises(Exception):summarize(rows[:-1])
        with self.assertRaises(Exception):summarize(rows[:-1]+rows[:1])

    def test_resume_rejects_wrong_split_tokens_and_candidate(self):
        args=('id',MODULES[0],'Marginal',0,'tokens','freeze')
        row=dict(identity='id',module=MODULES[0],method='Marginal',window=0,role='test',
                 token_hash='tokens',freeze_hash='freeze',scores=dict(tokens=2047,KL=.01))
        check_score(row,*args)
        for field,value in [('role','validation'),('token_hash','other'),('method','A-only'),('freeze_hash','other')]:
            with self.assertRaises(Exception):check_score(dict(row,**{field:value}),*args)


if __name__=='__main__':unittest.main()
