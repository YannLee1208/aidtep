from aidtep.inverse_problem import InverseBuilder


class IAEAInverseBuilder(InverseBuilder):
    @classmethod
    def name(cls):
        return "IAEA"
