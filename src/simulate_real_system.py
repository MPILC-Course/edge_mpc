import numpy as np


def simulate_real_system(y_past, u_past, u_current, step):
    # Updated ARX model parameters from the uploaded image
    a1 = 5.9998E-01  # Parameter for y(k-1)
    a2 = 4.8249E-02  # Parameter for y(k-2)
    b1 = 4.4071E-01  # Parameter for u(k-1)
    c = -2.8262E+02  # Constant parameter

    # Ensure y_past and u_past are numpy arrays
    y_past = np.asarray(y_past)
    u_past = np.asarray(u_past)

    # Calculate the next output using the ARX model, assuming the 'a3' and 'b2' parameters are not used
    y_next = a1 * y_past[-1] + a2 * y_past[-2] + b1 * u_past[-1] + c
    if step % 15 == 0:
        y_next += np.random.randint(2, 4)  # Add a random integer between 2 and 3

    # Update the lags for y and u
    updated_y_past = np.append(y_past[1:], y_next)  # Drop the oldest y value, append y_next
    updated_u_past = np.append(u_past[1:], u_current)  # Drop the oldest u value, append u_current

    return updated_y_past, updated_u_past, y_next