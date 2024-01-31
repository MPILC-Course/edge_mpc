import casadi as ca

class ARXModel:
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C
        self.arx_model = self.create_arx_model(A, B, C)

    def create_arx_model(self, A, B, C):
        A_power = ca.SX(A)
        B_power = ca.SX(B)
        C_power = ca.SX(C)

        y_past_power = ca.SX.sym('y_past_power', 2, 1)
        u_past_power = ca.SX.sym('u_past_power', 1, 1)

        y_power = ca.mtimes(A_power.T, y_past_power) + ca.mtimes(B_power, u_past_power) + C_power
        arx_model = ca.Function('arx_model', [y_past_power, u_past_power], [y_power])
        return arx_model

    def update_past_values(self, k, y, u, y_past_initial, u_past_initial):
        y_past = ca.vertcat(y[k], y[k-1]) if k > 0 else y_past_initial
        u_past = ca.vertcat(u[k-1]) if k > 0 else u_past_initial
        return y_past, u_past