import argparse
import sys
from logic import Store
import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go buy some mangoes.")
    parser.add_argument("N", type=int, help="max customer in store")
    parser.add_argument("M", type=int, help="max mango storage space")
    parser.add_argument("E1", type=float, help="cost: order & transport")
    parser.add_argument("E2", type=float, help="cost: per box")
    parser.add_argument("E3", type=float, help="cost: storage per hour")
    parser.add_argument("S", type=float, help="sale price per box")
    parser.add_argument("C", type=float, help="selling price per box")
    parser.add_argument("P", type=float, help="probability of customer buying a box of mangoes")
    parser.add_argument("probfuncname", choices=["normal", "poisson"], help="probability function to use")
    parser.add_argument("iteration_type", choices=["value", "policy"], help="select value or policy iteration, policy iteration needs -m and -t")
    parser.add_argument("-m", nargs=1, type=int, help="number of mangoes currently in stock")
    parser.add_argument("-t", nargs=1, type=int, help="current time of day")
    parser.add_argument("--printtimes", action="store_true", help="print the iteration function processing time")

    args = parser.parse_args()

    s = Store(args.N, args.M, args.E1, args.E2, args.E3, args.S, args.C, args.P, args.probfuncname)
    if (args.iteration_type == "value"):
        print("Running value iteration...")
        start = datetime.datetime.now()
        utilities = s.value_iteration()
        end = datetime.datetime.now()
        if args.printtimes: print((end-start).total_seconds())
        print("listing the utilities of every state.....")
        print(utilities)
    elif (args.iteration_type == "policy"):
        if args.m is None or args.t is None:
            print("optimal argument missing to policy iteration. See '--help'.")
            exit(1)
        print("Running policy iteration.....")
        start = datetime.datetime.now()
        best_action = s.get_best_from_policy_iteration(args.m, args.t)
        end = datetime.datetime.now()
        if args.printtimes: print((end-start).total_seconds())
        print(f"best action to take: order {best_action}")

exit(0)