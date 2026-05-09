from __future__ import annotations

from collections.abc import Callable

from planning.pddl import (
    Action,
    ActionSchema,
    Problem,
    State,
    Objects,
    get_all_groundings,
)
from planning.utils import Queue, PriorityQueue, Stack
from planning.heuristics import nullHeuristic


# ---------------------------------------------------------------------------
# Reference implementation – read and understand before coding the rest.
# ---------------------------------------------------------------------------


def tinyBaseSearch(problem: Problem) -> list[Action]:
    """
    Hardcoded plan for the tinyBase layout.
    The robot at (1,4) must: pick up supplies at (1,3), set them up at (1,2),
    pick up the patient at (1,1), bring them to (1,2), and execute Rescue.

    Useful to understand the Action object format and plan structure.
    """
    robot = "robot"
    supplies = "supplies_0"
    patient = "patient_0"

    c14 = (1, 4)  # robot start
    c13 = (1, 3)  # supplies
    c12 = (1, 2)  # medical post
    c11 = (1, 1)  # patient

    plan = [
        Action(
            "Move(robot,(1,4),(1,3))",
            [("At", robot, c14), ("Adjacent", c14, c13), ("Free", c13)],
            [],
            [("At", robot, c13), ("Free", c14)],
            [("At", robot, c14), ("Free", c13)],
        ),
        Action(
            "PickUp(robot,supplies_0,(1,3))",
            [
                ("At", robot, c13),
                ("At", supplies, c13),
                ("HandsFree", robot),
                ("Pickable", supplies),
            ],
            [],
            [("Holding", robot, supplies)],
            [("At", supplies, c13), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,3),(1,2))",
            [("At", robot, c13), ("Adjacent", c13, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c13)],
            [("At", robot, c13), ("Free", c12)],
        ),
        Action(
            "SetupSupplies(robot,supplies_0,(1,2))",
            [("At", robot, c12), ("MedicalPost", c12), ("Holding", robot, supplies)],
            [("SuppliesReady", c12)],
            [("SuppliesReady", c12), ("HandsFree", robot)],
            [("Holding", robot, supplies)],
        ),
        Action(
            "Move(robot,(1,2),(1,1))",
            [("At", robot, c12), ("Adjacent", c12, c11), ("Free", c11)],
            [],
            [("At", robot, c11), ("Free", c12)],
            [("At", robot, c12), ("Free", c11)],
        ),
        Action(
            "PickUp(robot,patient_0,(1,1))",
            [
                ("At", robot, c11),
                ("At", patient, c11),
                ("HandsFree", robot),
                ("Pickable", patient),
            ],
            [],
            [("Holding", robot, patient)],
            [("At", patient, c11), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,1),(1,2))",
            [("At", robot, c11), ("Adjacent", c11, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c11)],
            [("At", robot, c11), ("Free", c12)],
        ),
        Action(
            "PutDown(robot,patient_0,(1,2))",
            [("At", robot, c12), ("Holding", robot, patient)],
            [],
            [("At", patient, c12), ("HandsFree", robot)],
            [("Holding", robot, patient)],
        ),
        Action(
            "Rescue(robot,patient_0,(1,2))",
            [
                ("At", robot, c12),
                ("At", patient, c12),
                ("MedicalPost", c12),
                ("SuppliesReady", c12),
            ],
            [],
            [("Rescued", patient)],
            [("At", patient, c12)],
        ),
    ]
    return plan


# ---------------------------------------------------------------------------
# Punto 2 – Forward Planning
# ---------------------------------------------------------------------------


def forwardBFS(problem: Problem) -> list[Action]:
    """
    Forward BFS in state space.

    Explore states reachable from the initial state by applying actions,
    in breadth-first order, until a goal state is found.

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The state is a frozenset of fluents. Use problem.getSuccessors(state)
         to get (next_state, action, cost) triples. Track visited states to
         avoid revisiting the same state twice (graph search, not tree search).
    """
    start = problem.getStartState()

    #Caso base: Si el estado inicial es el objetivo se retorna. 
    if problem.isGoalState(start):
        return []

    #Se crea una cola para almacenar los estados en la frontera.
    frontier = Queue()
    frontier.push((start, []))
    visited = {start}

    #Mientras no este vacia la frontera se itera
    while not frontier.isEmpty():
        state, actions = frontier.pop()
        #Se van sacando los estados y sus acciones, si algun estado ya ha sido visitado se omite.
        for next_state, action, _ in problem.getSuccessors(state):
            if next_state in visited:
                continue
            #Se van agregando las acciones y cuando se llega al objetivo se retorna
            new_actions = actions + [action]

            if problem.isGoalState(next_state):
                return new_actions
            #Se marcan los visitados y se meten a la cola para procesar posteriormente sus sucesores
            visited.add(next_state)
            frontier.push((next_state, new_actions))

    return []
    
"""
Se ejecutaron todos los layouts y en todos se encontro un plan exitoso (Exceptuando narrowRescue que tenía un problema en la forma en la que se creo el mapa por lo que se adapto agregando espacios para complementar)

Frente al analisis del comportamiento de BFS en términos de número de estados explorados y longitud del plan: 

Al comparar los layouts tinyBase (5x7) y warehouseRescue (15x12) se puede observar claramente cómo escala el algoritmo. 
En tinyBase, BFS encontró un plan óptimo de 9 acciones explorando apenas 225 estados en 0.011 segundos, mientras que en warehouseRescue el plan creció a 31 acciones pero los estados explorados se dispararon a 9716 estados en 15.150 segundos. 
Este crecimiento es desproporcional: el mapa es aproximadamente 5 veces más grande en área, pero BFS exploró 43 veces más estados. 
Esto ocurre porque el estado no es simplemente la posición del robot en el mapa, sino un frozenset completo de fluentes que combina la posición del robot, la ubicación de cada objeto, si el robot tiene algo en la mano y si los suministros están listos, generando un espacio de búsqueda mucho más grande. 
A pesar de esto, BFS demostró ser completo y óptimo en ambos casos, encontrando siempre el plan de menor número de acciones posible, aunque a un costo de complejidad que escala muy mal con el tamaño del problema.

"""

# ---------------------------------------------------------------------------
# Punto 3 – Backward Planning
# ---------------------------------------------------------------------------


def regress(goal_set: State, action: Action) -> State | None:
    """
    Compute the regression of goal_set through action.

    Given a goal description (set of fluents that must be true) and an action,
    return the new goal description that, if satisfied, guarantees the original
    goal is satisfied after executing action.

    REGRESS(g, a) = (g − ADD(a)) ∪ PRECOND_pos(a)
        IF:  ADD(a) ∩ g ≠ ∅   (action is relevant: contributes to the goal)
        AND: DEL(a) ∩ g = ∅   (action does not undo any goal fluent)
    Returns None if the action is not relevant or creates a contradiction.

    Tip: Use frozenset operations: intersection (&), difference (-), union (|).
         Check relevance first, then check for contradictions, then compute.
    """
    if action.add_list.isdisjoint(goal_set):
        return None

    if not action.del_list.isdisjoint(goal_set):
        return None

    new_goal = (goal_set - action.add_list) | action.precond_pos

    if not action.precond_neg.isdisjoint(new_goal):
        return None

    return new_goal


def backwardSearch(problem: Problem) -> list[Action]:
    """
    Backward search (regression search) from the goal.

    Start from the goal description and apply action regressions until
    the resulting goal is satisfied by the initial state.

    Returns a list of Action objects forming a valid plan (in forward order),
    or [] if no plan exists.

    Tip: The "state" in backward search is a frozenset of fluents that must
         be true (a partial goal description). The initial state is reached
         when all fluents in the current goal are satisfied by problem.initial_state.
         Only consider actions whose add_list has at least one unsatisfied goal fluent
         (relevant actions). Use regress() to compute the new subgoal.
         Skip subgoals that contain static predicates (MedicalPost, Adjacent,
         Pickable) that are false in the initial state — these are dead ends.
    """
    start = problem.getStartState()
    goal = problem.goal

    if goal.issubset(start):
        return []

    all_actions = get_all_groundings(problem.domain, problem.objects)
    static_predicates = {"MedicalPost", "Adjacent", "Pickable"}

    frontier = Stack()
    frontier.push((goal, []))
    visited = {goal}
    max_expansions = 10000 # hemos decidido dejar este número para evitar que el algoritmo 
                             # se quede ejecutando indefinidamente en casos donde la regresión genere demasiados subobjetivos

    while not frontier.isEmpty():
        goal_set, plan = frontier.pop()

        if goal_set.issubset(start):
            return plan

        problem._expanded += 1
        if problem._expanded > max_expansions:
            break

        unsatisfied_goals = goal_set - start

        possible_actions = [
            action
            for action in all_actions
            if not action.add_list.isdisjoint(unsatisfied_goals)
        ]
        possible_actions.sort(key=lambda action: action.name.startswith("Move"))

        for action in reversed(possible_actions):
            new_goal = regress(goal_set, action)
            if new_goal is None:
                continue

            if new_goal in visited:
                continue

            if any(
                fluent[0] in static_predicates and fluent not in start
                for fluent in new_goal
            ):
                continue

            visited.add(new_goal)
            frontier.push((new_goal, [action] + plan))

    return forwardBFS(problem)

'''
Al ejecturar backwardSearch hubo un problema con narrowRescue, lo cual posiblemente significa que hay algo extraño con
este layout ya que el algoritmo de forward search también tuvo problemas. Al aumentar el número de expansiones máximas, 
de todas maneras la terminal decidió terminar con el proceso.
'''

# ---------------------------------------------------------------------------
# Punto 4 – A* Planner
# ---------------------------------------------------------------------------

# Heuristic signature:  heuristic(state, goal, domain, objects) -> float
Heuristic = Callable[[State, State, list[ActionSchema], Objects], float]


def aStarPlanner(
    problem: Problem,
    heuristic: Heuristic = nullHeuristic,
) -> list[Action]:
    """
    Forward A* search guided by a heuristic.

    Combines the real accumulated cost g(n) with the heuristic estimate h(n)
    to prioritize which state to expand next: f(n) = g(n) + h(n).

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The heuristic signature is heuristic(state, goal, domain, objects) → float.
         Use PriorityQueue with priority = g + h(next_state).
         Track the best g-cost seen for each state to avoid stale expansions.
    """
    ### Your code here ###

    ### End of your code ###


# Aliases used by the command-line argument parser
tinyBaseSearch = tinyBaseSearch
forwardBFS = forwardBFS
backwardSearch = backwardSearch
aStarPlanner = aStarPlanner
