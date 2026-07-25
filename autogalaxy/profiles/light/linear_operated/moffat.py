from autogalaxy.profiles.light import operated as lp_operated
from autogalaxy.profiles.light import linear as lp_linear


class Moffat(lp_linear.Moffat, lp_operated.LightProfileOperated):
    pass
