from __future__ import annotations
from collections import deque

from planning.pddl import Action, Problem, apply_action, is_applicable


# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.

    name:        Human-readable name for display
    refinements: List of possible refinements, each a list of HLA/Action objects
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """Return True if action is a primitive (grounded Action), False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """Return True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------
def simulate_plan(remaining, state):
    # si no quedan acciones es porque el plan fue válido
    if not remaining:
        return True, state
    action = remaining[0]
    # Si la acción no se puede hacer desde el estado actual entonces el plan es inválido
    if not is_applicable(state, action):
        return False, None
    # Hacer la acción y seguir con el resto del plan
    return simulate_plan(remaining[1:], apply_action(state, action))


def find_first_hla(plan):
    # Buscar índice de primera HLA no primitiva
    indices = []
    for i, step in enumerate(plan):
        if not is_primitive(step):
            indices.append(i)
    if indices:
        return indices[0]
    return None

def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:
    """
    HTN planning via BFS over hierarchical plan refinements.

    Start with an initial plan containing a single top-level HLA.
    At each step, find the first non-primitive step in the plan and
    replace it with one of its refinements. Continue until the plan
    is fully primitive and achieves the goal when executed from the
    initial state.

    Returns a list of primitive Action objects, or [] if no plan found.

    Tip: The search space consists of (partial plan, current plan index) pairs.
         Use a Queue (BFS) to explore all refinement choices fairly.
         A plan is a solution when:
           1. It contains only primitive actions (is_plan_primitive), AND
           2. Executing it from the initial state reaches a goal state.
         To simulate execution, apply each action in order using apply_action().
    """
    #solo la HLA raíz (plan inicial)
    frontera = deque([[hlas[0]]])
    
    while frontera:
        plan = frontera.popleft()

        if is_plan_primitive(plan):
            # si el plan es completamente primitivo etnonces se simula y verifica si alcanza el objetivo
            valid, final_state = simulate_plan(plan, problem.initial_state)
            if valid and problem.isGoalState(final_state):
                return plan
        else:
            # encontrar la primera HLA y reemplazarla por cada una de sus subacciones
            first_hla_idx = find_first_hla(plan)
            if first_hla_idx is not None:
                hla = plan[first_hla_idx]
                for refinement in hla.refinements:
                    # ahora el nuevo plan es el anterior con la hla cambiada por sus subacciones
                    new_plan = plan[:first_hla_idx] + refinement + plan[first_hla_idx + 1:]
                    frontera.append(new_plan)

    return []

    



# ---------------------------------------------------------------------------
# Punto 5b – HLA Definitions
# ---------------------------------------------------------------------------
def get_medical_post(i, medical_posts):
    # si hay menos puestos que misiones entonces se debe reusar el primero
    if i < len(medical_posts):
        return medical_posts[i]
    return medical_posts[0]


def bfs(start, goal, adjacent_pairs):
    # Si ya está en el destino entonces no se hace nada
    if start == goal:
        return [start]

    # construir grafo de adyacentes desde el estado inicial
    graph = {}
    for a, b in adjacent_pairs:
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []
        graph[a].append(b)
        graph[b].append(a)

    frontier = deque([(start, [start])])
    visited = {start}
    result = []

    # se hace BFS hasta encontrar el destino o que no haya más nodos
    while frontier and not result:
        current, path = frontier.popleft()
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                if neighbor == goal:
                    # si se encuentra el destino entonces se devuelve el camino
                    result = path + [neighbor]
                else:
                    # añadir a la frontera los vecinos no visitados
                    visited.add(neighbor)
                    frontier.append((neighbor, path + [neighbor]))

    if result:
        return result
    return [start]


def path_to_moves(path, robot):
    # convierte una lista de celdas en acciones Move
    moves = []
    for i in range(len(path) - 1):
        from_cell = path[i]
        to_cell = path[i + 1]
        moves.append(Action(
            f"Move({robot},{from_cell},{to_cell})",
            precond_pos=[
                ("At", robot, from_cell),
                ("Adjacent", from_cell, to_cell),
                ("Free", to_cell),
            ],
            precond_neg=[],
            add_list=[("At", robot, to_cell), ("Free", from_cell)],
            del_list=[("At", robot, from_cell), ("Free", to_cell)],
        ))
    return moves


def make_pickup(obj, loc, robot):
    # recoger un objeto en una celda
    return Action(
        f"PickUp({robot},{obj},{loc})",
        precond_pos=[
            ("At", robot, loc),
            ("At", obj, loc),
            ("HandsFree", robot),
            ("Pickable", obj),
        ],
        precond_neg=[],
        add_list=[("Holding", robot, obj)],
        del_list=[("At", obj, loc), ("HandsFree", robot)],
    )


def make_putdown(obj, loc, robot):
    # dejar un objeto en una celda
    return Action(
        f"PutDown({robot},{obj},{loc})",
        precond_pos=[
            ("At", robot, loc),
            ("Holding", robot, obj),
        ],
        precond_neg=[],
        add_list=[("At", obj, loc), ("HandsFree", robot)],
        del_list=[("Holding", robot, obj)],
    )


def make_setup(s, loc, robot):
    # poner suministros en el puesto médico
    return Action(
        f"SetupSupplies({robot},{s},{loc})",
        precond_pos=[
            ("At", robot, loc),
            ("MedicalPost", loc),
            ("Holding", robot, s),
        ],
        precond_neg=[("SuppliesReady", loc)],
        add_list=[("SuppliesReady", loc), ("HandsFree", robot)],
        del_list=[("Holding", robot, s)],
    )


def make_rescue(p, loc, robot):
    # rescatar el paciente en el puesto médico
    return Action(
        f"Rescue({robot},{p},{loc})",
        precond_pos=[
            ("At", robot, loc),
            ("At", p, loc),
            ("MedicalPost", loc),
            ("SuppliesReady", loc),
        ],
        precond_neg=[],
        add_list=[("Rescued", p)],
        del_list=[("At", p, loc)],
    )
    

def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    """
    Build HTN HLAs for the rescue domain.

    The hierarchy defines four HLA types:
      - Navigate(from, to):       Move the robot step by step from one cell to another
      - PrepareSupplies(s, m):    Collect supplies and set them up at the medical post
      - ExtractPatient(p, m):     Pick up the patient and bring them to the medical post
      - FullRescueMission(s,p,m): Complete one rescue: prepare supplies + extract + rescue

    Refinements are built from the ground state to generate concrete Action objects.

    Tip: Refinements for Navigate are all single-step Move sequences between
         adjacent cells. PrepareSupplies and ExtractPatient chain Navigate HLAs
         with primitive PickUp, SetupSupplies, PutDown, and Rescue actions.
    """
    robot = "robot"
    objects = problem.objects
    initial_state = problem.initial_state

    supplies_list = objects["supplies"]
    patients_list = objects["patients"]
    medical_posts = objects["medical_posts"]

    # sacar pares adyacentes del estado inicial para construir el grafo del mapa
    adjacent_pairs = []
    for f in initial_state:
        if f[0] == "Adjacent":
            adjacent_pairs.append((f[1], f[2]))

    # posición inicial del robot
    robot_start = None
    for f in initial_state:
        if f[0] == "At" and f[1] == robot:
            robot_start = f[2]

    # posiciones iniciales de suministros
    supplies_pos = {}
    for s in supplies_list:
        for f in initial_state:
            if f[0] == "At" and f[1] == s:
                supplies_pos[s] = f[2]

    # posiciones iniciales de pacientes
    patients_pos = {}
    for p in patients_list:
        for f in initial_state:
            if f[0] == "At" and f[1] == p:
                patients_pos[p] = f[2]

    full_mission_hlas = []
    # miramos que puestos ya tienen suministros puestos entre misiones
    sim_supplies_ready = set()
    robot_pos = robot_start
    i = 0

    for patient, supplies in zip(patients_list, supplies_list):
        medical_post = get_medical_post(i, medical_posts)
        s_pos = supplies_pos[supplies]
        p_pos = patients_pos[patient]

        mission_actions = []

        if medical_post not in sim_supplies_ready:
            # navegar a suministros, recoger, ir al puesto y instalar
            mission_actions += path_to_moves(bfs(robot_pos, s_pos, adjacent_pairs), robot)
            mission_actions += [make_pickup(supplies, s_pos, robot)]
            mission_actions += path_to_moves(bfs(s_pos, medical_post, adjacent_pairs), robot)
            mission_actions += [make_setup(supplies, medical_post, robot)]
            sim_supplies_ready.add(medical_post)
        else:
            # el puesto ya tiene suministros entonces solo se debe mover el robot hasta allá
            mission_actions += path_to_moves(bfs(robot_pos, medical_post, adjacent_pairs), robot)

        # navegar al paciente, recoger,  ir al puesto,  dejar y rescatar
        mission_actions += path_to_moves(bfs(medical_post, p_pos, adjacent_pairs), robot)
        mission_actions += [make_pickup(patient, p_pos, robot)]
        mission_actions += path_to_moves(bfs(p_pos, medical_post, adjacent_pairs), robot)
        mission_actions += [make_putdown(patient, medical_post, robot)]
        mission_actions += [make_rescue(patient, medical_post, robot)]

        # después de cada misión el robot queda en el puesto médico
        robot_pos = medical_post

        full_mission_hlas.append(HLA(
            f"FullRescueMission({supplies},{patient},{medical_post})",
            refinements=[mission_actions],
        ))

        i += 1

    # una misión simple o todas encadenadas para multi-rescate
    if len(full_mission_hlas) == 1:
        root = full_mission_hlas[0]
    else:
        # unir todas las misiones en una sola subaccion secuencial
        combined = []
        for fm in full_mission_hlas:
            combined.extend(fm.refinements[0])
        root = HLA("AllRescueMissions", refinements=[combined])

    return [root]

