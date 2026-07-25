from autogalaxy.profiles.light import standard as lp

from autogalaxy.profiles.light.operated.abstract import LightProfileOperated


class Moffat(lp.Moffat, LightProfileOperated):
    pass
