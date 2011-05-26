import unittest
import Image
import imfeat
import hashlib
import numpy as np
import scipy as sp
import scipy.io
import cv


class TestFeatures(unittest.TestCase):

    def setUp(self):
        self.image_names = ['test_images/test1.jpg',
                            'test_images/test2.jpg']

    def _histogram(self, hf):
        # Test the histogram module by loading each test image
        # and converting it to each possible mode.
        # Also check that the size of the returned histogram is as
        # exepcted.
        for fn in self.image_names:
            img = Image.open(fn)
            h = imfeat.compute(hf, img)
            self.assertTrue(len(h) == 1)
            self.assertTrue(h[0].shape == (8*8*8,))

    def _run_all_images(self, feature):
        images = (Image.open(fn)
                  for fn in self.image_names)
        return ((imfeat.compute(feature, image), image)
                for image in images)

    def test_histogram_joint(self):
        self._histogram(imfeat.Histogram('rgb'))

    def test_histogram_joint_lab(self):
        self._histogram(imfeat.Histogram('lab'))

    def test_moments(self):
        feature = imfeat.Moments('rgb', 2)
        for feat_out, image in self._run_all_images(feature):
            image = np.asfarray(image) / 255.
            image = image.reshape((image.shape[0] * image.shape[1],
                                   image.shape[2]))
            np.testing.assert_almost_equal(np.mean(image, 0), feat_out[0][:3])
            print(np.var(image, 0), feat_out[0][3:6])
            np.testing.assert_almost_equal(np.var(image, 0), feat_out[0][3:6])

    def test_rhog_gray(self):
        print('RHOG_GRAY')
        feature = imfeat.RHOG()
        for feat_out, image in self._run_all_images(feature):
            print(feat_out)
            print(len(feat_out[0]))

    def test_rhog_rgb(self):
        feature = imfeat.RHOG(gray=False)
        for feat_out, image in self._run_all_images(feature):
            print(feat_out)
            print(len(feat_out[0]))

    def test_surf_random(self):
        feature = imfeat.SURF()
        for feat_out, image in self._run_all_images(feature):
            print(feat_out)
            print(len(feat_out[0]))

    def test_surf_detect(self):
        feature = imfeat.SURF(method='detect')
        for feat_out, image in self._run_all_images(feature):
            print(feat_out)
            print(len(feat_out[0]))

    def test_hog_latent(self):
        feature = imfeat.HOGLatent(2)
        for feat_out, image in self._run_all_images(feature):
            print(feat_out)
            print(len(feat_out[0]))
        print('Hog Latent')
        image = Image.open('test_images/lena.ppm')
        out = imfeat.compute(feature, image)[0]
        self.assertEqual(len(out), 254 * 254 * 32)
        np.testing.assert_equal(hashlib.md5(out).hexdigest(),
                                '18231ad8c359860ee09e8f0fa8b316a1')

    def test_gist(self):
        print 'GIST'
        feature = imfeat.GIST()
        for feat_out, image in self._run_all_images(feature):
            #print(feat_out)
            #print(len(feat_out[0]))
            pass
        # Compare against known output
        image = Image.open('test_images/lena.ppm')
        out = imfeat.compute(feature, image)[0]
        test_string = ' '.join(['%.4f' % x for x in out] + ['\n'])
        with open('fixtures/gist_lena_output.txt') as fp:
            true_string = fp.read()
        self.assertEqual(len(true_string.split()), len(test_string.split()))
        self.assertEqual(true_string, test_string)

    def _test_feature(self, name, feature):
        image = Image.open(name)
        out = imfeat.compute(feature, image)[0]
        return out

    def test_moments_large(self):
        self._test_feature('test_images/large.jpg', imfeat.Moments('rgb', 2))

    @unittest.skip("GIST large segfaults")
    def test_gist_large(self):
        self._test_feature('test_images/large.jpg', imfeat.GIST())

    def test_hog_large(self):
        self._test_feature('test_images/large.jpg', imfeat.HOGLatent(2))

    def test_gist_small(self):
        image = cv.LoadImage('test_images/small.png')
        feature = imfeat.GIST()
        out = imfeat.compute(feature, image)[0]
        return out

if __name__ == '__main__':
    unittest.main()
