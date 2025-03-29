
t = 0
t_0 = 1.26
t_int = 4.0 * t_0 + t
t_emk = 1.43 * t_0
t_a = 4.46 * t_0
tau = t_emk + t

k = 1.0 / (1.552 * tau / t_int + 0.078)
t_i = t_int * ((0.186 * tau / t_int) + 0.532)
t_d = 0.25 * t_i
t_s = t_d / 8.0

# k = 1.0 / (2.766 * tau / t_int - 0.521)
# t_i = t_int * ((-0.15 * tau / t_int) + 0.552)
# t_d = 0.4 * t_i
# t_s = t_d / 8.0

print("k = " + str(k))
print("t_i = " + str(t_i))
print("t_d = " + str(t_d))
print("t_s = " + str(t_s))
