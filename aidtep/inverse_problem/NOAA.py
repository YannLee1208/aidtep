from aidtep.inverse_problem import InverseBuilder


class NOAAInverseBuilder(InverseBuilder):
    @classmethod
    def name(cls):
        return "NOAA"
