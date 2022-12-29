from ctypes import util
from decimal import MAX_EMAX
import math
import numpy as np

class Store:
    def __init__(self, N:int, M:int, E1:float, E2:float, E3:float, S:float, C:float, P:float, probfuncname) -> None:
        self.N = N
        self.M = M
        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.S = S
        self.C = C
        self.P = P
        if probfuncname == "normal":
            self.f = self.normal

        elif probfuncname == "poisson":
            self.f = self.poisson

        self.discount_factor = 0.9
        self.maximum_error = 0.1

    def transition_prob(self, m_new, m_old, t_old, o_a):
        if o_a > m_new:
            return 0
        k = m_old + o_a - m_new
        sum = 0
        for n in range(self.N + 1):
            sum += self.f(n,t_old) * self.Q(n,k)
        return sum
    
    def reward(self, m_new, m_old, o_a):
        k = m_old + o_a - m_new
        return (
            self.S*k
            - (self.E1 if o_a > 0 else 0)
            - self.E2 * o_a
            - self.E3 * (m_old - k)
        )
    
    def q_value(self, m_old, t_old, o_a, utilities):
        t_new = 0 if t_old == 24 else t_old + 1
        sum = 0
        for m_new in range(self.M + 1):
            sum += self.transition_prob(m_new, m_old, t_old, o_a) * (
                self.reward(m_new, m_old, o_a)
                +
                (self.discount_factor*utilities[m_new, t_new])
            )
        return sum

    def value_iteration(self):
        utilities = np.zeros((self.M+1, 24))
        next_utilities = np.zeros((self.M+1, 24))
        absolute_maximum_error = self.maximum_error * (1-self.discount_factor) / self.discount_factor
        while True:
            utilities = np.copy(next_utilities)
            delta = 0
            for m_old in range(self.M+1):
                for t_old in range(24):
                    max_q_value = -1000000
                    for o_a in range(self.M+1):
                        max_q_value = max(max_q_value, self.q_value(m_old, t_old, o_a, utilities))

                    next_utilities[m_old, t_old] = max_q_value
                    abs_new_delta = abs(next_utilities[m_old, t_old] - utilities[m_old, t_old])
                    if abs_new_delta > delta:
                        delta = abs_new_delta
            if delta <= absolute_maximum_error: break

        return utilities

    def policy_evaluation(self, policy, utilities):
        new_utilities = np.zeros((self.M, 24))
        for m_old in range(self.M+1):
            for t_old in range(24):
                action = policy[m_old, t_old]
                new_utilities[m_old, t_old] = self.q_value(m_old, t_old, action, utilities)

            return new_utilities

    def policy_iteration(self):
        utilities = np.zeros((self.M, 24))
        policy = np.zeros((self.M+1,24))

        while True:
            utilities = self.policy_evaluation(policy, utilities)
            is_unchanged = True

            for m_old in range(self.M+1):
                for t_old in range(24):
                    max_q_value = -1000000
                    for o_a in range(self.M+1):
                        new_q_value = self.q_value(m_old, t_old, o_a, utilities)
                        if (new_q_value > max_q_value):
                            max_q_value = new_q_value
                            best_action = o_a

                        if (
                            self.q_value(m_old, t_old, best_action, utilities)
                            >
                            self.q_value(m_old, t_old, policy[m_old, t_old], utilities)
                        ):
                            policy[m_old, t_old] = best_action
                            is_unchanged = False

            if is_unchanged: break

        return policy

    def get_best_from_policy_iteration(self, m, t):
        policy = self.policy_iteration()
        return int(policy[m,t])

    def poisson(self, n, t):
        lam = (9/self.S) * self.g(t)
        prob = (
            math.exp(((-lam)) * (lam**n))
            /
            (math.factorial(n))
        )
        return 0 if prob < 0.001 else prob


    def normal(self, n, t):
        mu = (9/ self.S) * g(t)
        sigma = 3/ self.S
        prob = (
            math.exp(-((n-mu)*(n-mu))/(2*sigma*sigma))
            /
            (sigma * math.sqrt(2*math.pi))
        )
        return 0 if prob < 0.001 else prob

    def Q(self, n, k):
        if (k > n): return 0
        if (k < 0): return 0
        return (
            (self.P ** k)
            * ((1-self.P) ** (n-k))
            * math.factorial(n)
            / (math.factorial(k) * math.factorial(n-k))
        )
    
    def g(self, t):
        return 1-((t-12)*(t-12)/144)

if __name__ == "__main__":
    s = Store(20, 10, 100, 2, 0.5, 5, 1, 0.5, "normal")
    for n in range(20):
        print(s.f(n,23))