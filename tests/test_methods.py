"""Gezielte mathematische Prüfungen der für die Interpretation wichtigen Schritte."""
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from analyse import normalized_factors, topic_match, relative_error, basic_clean


class MethodChecks(unittest.TestCase):
    def test_nmf_scaling_preserves_reconstruction_and_dominance(self):
        w=np.array([[2.,1.],[.5,3.]])
        h=np.array([[1.,1.,2.],[4.,0.,0.]])
        wn,hn=normalized_factors(w,h)
        np.testing.assert_allclose(wn@hn,w@h)
        np.testing.assert_allclose(hn.sum(1),1)
        # Beliebige reziproke Umskalierung desselben Modells darf das Ergebnis nicht ändern.
        a,b=normalized_factors(w/np.array([10.,.1]),h*np.array([10.,.1])[:,None])
        np.testing.assert_allclose(a,wn)
        np.testing.assert_array_equal(a.argmax(1),wn.argmax(1))

    def test_topic_matching_ignores_arbitrary_topic_numbers(self):
        a=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        self.assertAlmostEqual(topic_match(a,a[[2,0,1]]),1.0)

    def test_sparse_reconstruction_error_matches_direct_calculation(self):
        rng=np.random.RandomState(2)
        x=rng.rand(5,8);w=rng.rand(5,3);h=rng.rand(3,8)
        expected=np.linalg.norm(x-w@h)/np.linalg.norm(x)
        self.assertAlmostEqual(relative_error(csr_matrix(x),w,h),expected)

    def test_cleaning_keeps_negation_and_removes_redaction_artifacts(self):
        text="I did not owe XXXX 123 dollars. https://example.org jane@example.org"
        clean=basic_clean(text)
        self.assertIn("not owe",clean)
        self.assertNotIn("XXXX",clean)
        self.assertNotIn("example",clean)


if __name__=='__main__': unittest.main()
