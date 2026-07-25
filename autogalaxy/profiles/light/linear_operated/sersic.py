from autogalaxy.profiles.light import operated as lp_operated
from autogalaxy.profiles.light import linear as lp_linear


class Sersic(lp_linear.Sersic, lp_operated.LightProfileOperated):
    pass
