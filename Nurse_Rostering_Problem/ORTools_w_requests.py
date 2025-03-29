"""Example of a simple nurse scheduling problem."""
from typing import Union
from ortools.sat.python import cp_model

shift_names = ["Morning", "Evening", "Night"]

def main() -> None:
    # Data.
    num_nurses = 5
    num_shifts = 3
    num_days = 7
    all_nurses = range(num_nurses)
    all_shifts = range(num_shifts)
    all_days = range(num_days)
    shift_requests = [
        [[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]],
        [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1]],
        [[0, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
    ]
    day_off_requests = [
        [0, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 0],
    ]

    # Creates the model.
    model = cp_model.CpModel()

    # Creates shift variables.
    shifts = {}
    for n in all_nurses:
        for d in all_days:
            for s in all_shifts:
                shifts[(n, d, s)] = model.new_bool_var(f"shift_n{n}_d{d}_s{s}")

    # Each shift is assigned to exactly one nurse.
    for d in all_days:
        for s in all_shifts:
            model.add_exactly_one(shifts[(n, d, s)] for n in all_nurses)

    # Each nurse works at most one shift per day.
    for n in all_nurses:
        for d in all_days:
            model.add_at_most_one(shifts[(n, d, s)] for s in all_shifts)

    # Ensure fair distribution of shifts.
    min_shifts_per_nurse = (num_shifts * num_days) // num_nurses
    max_shifts_per_nurse = min_shifts_per_nurse + (1 if (num_shifts * num_days) % num_nurses != 0 else 0)

    for n in all_nurses:
        num_shifts_worked = sum(shifts[(n, d, s)] for d in all_days for s in all_shifts)
        model.add(min_shifts_per_nurse <= num_shifts_worked)
        model.add(num_shifts_worked <= max_shifts_per_nurse)

    # Maximize the number of respected shift requests.
    model.maximize(
        sum(
            shift_requests[n][d][s] * shifts[(n, d, s)]
            for n in all_nurses
            for d in all_days
            for s in all_shifts
        )+
        sum(
            (1 - sum(shifts[(n, d, s)] for s in all_shifts)) * day_off_requests[n][d]  # Reward respected day-off requests
            for n in all_nurses
            for d in all_days
        )
    )

    # Creates the solver.
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True  # Enumerate multiple solutions


    class SolutionPrinter(cp_model.CpSolverSolutionCallback):
        def __init__(self, shifts, num_nurses, num_days, num_shifts):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self._shifts = shifts
            self._num_nurses = num_nurses
            self._num_days = num_days
            self._num_shifts = num_shifts
            self._solution_count = 0
            self._best_objective = float('-inf')
            self._best_solutions = []

        def on_solution_callback(self):
            self._solution_count += 1
            current_objective = self.ObjectiveValue()

            # If this is a better solution, reset best solutions list
            if current_objective > self._best_objective:
                self._best_objective = current_objective
                self._best_solutions = [self.extract_solution()]
            elif current_objective == self._best_objective:
                self._best_solutions.append(self.extract_solution())

        def extract_solution(self):
            """Extracts the solution as a structured format."""
            solution = []
            for d in range(self._num_days):
                day_schedule = []
                for n in range(self._num_nurses):
                    shifts_worked = []
                    for s in range(self._num_shifts):
                        if self.value(self._shifts[(n, d, s)]):
                            shifts_worked.append(shift_names[s])
                    day_schedule.append((n, shifts_worked))
                solution.append(day_schedule)
            return solution

        def print_best_solutions(self):
            print(f"\nBest Objective Value: {self._best_objective}")
            print(f"Total optimal solutions found: {len(self._best_solutions)}")

            for i, sol in enumerate(self._best_solutions):
                print(f"\nOptimal Solution {i + 1}:")
                respected_requests = 0
                total_requests = sum(sum(sum(day) for day in nurse) for nurse in shift_requests) + sum(sum(day) for day in day_off_requests)

                for d, day_schedule in enumerate(sol):
                    print(f"Day {d + 1}:")
                    for nurse, shifts in day_schedule:
                        shift_str = ', '.join(shifts) if shifts else "No shift"
                        requested = any(shift_requests[nurse][d][s] == 1 for s in all_shifts if shift_names[s] in shifts) or day_off_requests[nurse][d] == 1
                        if requested:
                            respected_requests += 1
                        print(f"  Nurse {nurse}: {shift_str} {'(requested)' if requested else '(not requested)'}.")

                    print()
                print(f"Total respected requests: {respected_requests} / {total_requests}")

        def solution_count(self):
            return self._solution_count

    solution_printer = SolutionPrinter(shifts, num_nurses, num_days, num_shifts)

    # Solve and collect solutions.
    solver.solve(model, solution_printer)

    # Print all optimal solutions.
    solution_printer.print_best_solutions()

    # Statistics.
    print("\nStatistics")
    print(f"  - Conflicts: {solver.num_conflicts}")
    print(f"  - Branches : {solver.num_branches}")
    print(f"  - Wall time: {solver.wall_time}s")
    print(f"  - Solutions found: {solution_printer.solution_count()}")

if __name__ == "__main__":
    main()
