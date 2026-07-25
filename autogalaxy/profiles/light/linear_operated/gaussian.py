from autogalaxy.profiles.light import operated as lp_operated
from autogalaxy.profiles.light import linear as lp_linear


class Gaussian(lp_linear.Gaussian, lp_operated.LightProfileOperated):
    pass
