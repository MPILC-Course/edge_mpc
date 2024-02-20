import casadi as ca

class ARXModel:
    def __init__(self, A, B, C,y0,u0):
        self.A = A
        self.B = B
        self.C = C
        self.y_k = y0;
        self.u_k = u0;

    def update_state(self,u):
        y = self.A @ self.y_k +self.B @ self.u_k + self.C
        
        y_k_old = self.y_k[:-self.A.shape[0]] # Remove the oldest ny elements
        self.y_k = ca.vcat([y,y_k_old])

        u_k_old = self.u_k[:-u.shape[0]] # Remove the oldest nu elements
        self.u_k = ca.vcat([u,u_k_old])

        return y