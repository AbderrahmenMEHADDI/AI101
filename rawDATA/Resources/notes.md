### **Agent Hierarchy (Building on the Previous One)**

1️⃣ **Simple Reflex Agent**

A = f(P)

✔ **Base agent** → Only reacts to **current percept** P.

---

2️⃣ **Model-Based Reflex Agent** = **Simple Reflex Agent** + **Memory (State Tracking)**

A = f(P, S)

✔ Keeps **internal state** S (remembers past percepts).

---

3️⃣ **Goal-Based Agent** = **Model-Based Reflex Agent** + **Future Prediction**

A = f(P, S, G)

✔ Chooses actions **not just based on the present**, but based on what helps reach a **goal** G.

---

4️⃣ **Utility-Based Agent** = **Goal-Based Agent** + **Best-Action Selection**

A= arg⁡max⁡a U(f(P,S,G,a))

✔ **Not just reaching the goal**, but **finding the best way** using a **Utility Function** U.

---

5️⃣ **Learning Agent** = **Utility-Based Agent** + **Experience-Based Improvement**

A = f(P, S, G, U, L)

✔ Learns from past experiences L and **gets smarter over time**.

🔥 **Final Summary:**

- **Simple Reflex** → Reacts.
- **Model-Based Reflex** → Reacts + Remembers.
- **Goal-Based** → Remembers + Thinks ahead.
- **Utility-Based** → Thinks ahead + Picks the best.
- **Learning** → Evolves over time.

🚀 **Learning Agents = Ultimate Intelligence!** 🔥

---

### **Formal Definition of a Problem 

A **problem** in AI is defined by:

1️⃣ **Initial State** → Where the agent **starts**.  
2️⃣ **Actions** → What the agent **can do** in each state.  
3️⃣ **Transition Model** → What **happens** when an action is taken (how states change).  
4️⃣ **State Space** → **All possible states** reachable from the start.  
5️⃣ **Goal Test** → A check to see if the agent **has won** (reached the goal).  
6️⃣ **Path Cost Function** → **How expensive** a path is (lower cost = better solution).

✔ **A solution** = A sequence of actions that takes you from the **initial state** to the **goal state**.  
✔ **An optimal solution** = The one with the **lowest path cost**.

🔥 **In short:** Define where you start, what you can do, what happens, where you wanna go, and how good a solution is! 🚀


**Python Implementation:**
class Problem:
    """Abstract class for a formal problem. 
    Subclass this and implement actions, result, and possibly __init__, 
    goal_test, and path_cost to define a specific problem."""

    def __init__(self, initial, goal=None):
        """Initialize the problem with an initial state and an optional goal state."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the list of possible actions from a given state."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the resulting state after applying an action to a given state."""
        raise NotImplementedError

    def goal_test(self, state):
        """Check if the given state is a goal state."""
        return state in self.goal if isinstance(self.goal, list) else state == self.goal

    def path_cost(self, cost, state1, action, state2):
        """Return the cost of reaching state2 from state1 via action."""
        return cost + 1  # Default cost is 1 per step


### **Searching for Solutions

✔ **Solution** = Action sequence from **Start State → Goal State**.  
✔ **Search Tree** = All possible action sequences starting from the root (initial state).  
✔ **Nodes** = States, **Branches** = Actions.

---

### **How Search Works?**

1️⃣ Start at **initial state** (root).  
2️⃣ Expand the **current state** → Generate new states by applying all possible actions.  
3️⃣ **Key Decision** → Which child node do we expand next?

---

### **Important Terms:**

🔥 **Frontier** = Set of **leaf nodes** (not expanded yet).  
🔥 **Loopy Path** = A path that **repeats states** (wasting time going in circles).  
🔥 **Redundant Paths** = Different paths leading to the **same state**.

---

### **How to Avoid Loops?**

- **Path costs are additive** (each step adds cost).
- **Step costs are nonnegative** (no negative cycles).  
    ✔ So if we track visited states, we **avoid looping forever**! 🚀

### **Tree Search vs. Graph Search 

#### **🌳 Tree Search** (Basic but Wasteful)

- Starts from **initial state** and explores all possible action sequences.
- **No memory** → It can **revisit the same state many times**, causing loops.
- Good for exploring **small** state spaces but **inefficient for large ones**.

**Algorithm (Tree Search in Simple Steps)**  
1️⃣ Put the initial state in the **frontier** (list of unexplored nodes).  
2️⃣ While the **frontier is not empty**:

- Remove a node from the frontier.
- If it's the **goal state**, return the solution! 🎯
- Otherwise, **expand the node** → Add new states to the frontier.  
    3️⃣ If no solution is found, **return failure**.

---

#### **🌍 Graph Search** (Smarter & Efficient)

- Like Tree Search but **tracks visited states** (avoids revisiting them).
- **Prevents loops** and speeds up search in large graphs.
- Uses an **explored set** (stores visited nodes) to avoid redundancy.

**Algorithm (Graph Search in Simple Steps)**  
1️⃣ Put the initial state in the **frontier** and mark it as **visited**.  
2️⃣ While the **frontier is not empty**:

- Remove a node from the frontier.
- If it's the **goal state**, return the solution! 🎯
- Otherwise, **expand the node** → Add new states to the frontier **only if they haven't been visited**.
- Mark all new states as **visited**.  
    3️⃣ If no solution is found, **return failure**.

---

### **Key Difference:**

🔥 **Tree Search** = Searches blindly, might **revisit the same state** multiple times.  
🔥 **Graph Search** = **Tracks visited states**, avoids loops, and is more **efficient**.

✔ **Use Graph Search when dealing with large or cyclic problems!** 🚀

Every **node** in the search tree should have these four components:

1️⃣ **n.STATE** → The actual state in the problem space (where the agent is now).  
2️⃣ **n.PARENT** → The node that led to this one.  
3️⃣ **n.ACTION** → The action taken to move from **PARENT → this node**.  
4️⃣ **n.PATHCOST** (**g(n)**) → The cost from the **initial state** to this node (sum of step costs).

✔ **This structure is essential for reconstructing the solution path once we reach the goal!** 🚀

### **Problem-Solving Performance

🔹 **Completeness** → Will it always find a solution if one exists?  
🔹 **Optimality** → Will it find the **best** solution?  
🔹 **Time Complexity** → **How many nodes** does it generate?  
🔹 **Space Complexity** → **How much memory** does it use?

### **Search Complexity Factors:**

🔸 **b** → Branching factor (max successors per node).  
🔸 **d** → Depth of the **closest** goal node.  
🔸 **m** → Max depth of any path in the state space.

### **Costs:**

💰 **Search Cost** → Depends on **time complexity**.  
💰 **Path Cost** → Cost of the **solution** itself.  
💰 **Total Cost** = **Search Cost + Path Cost** (Might need a function to convert and sum).

✔ **The lower the cost, the better the algorithm! 🔥**

### **Uninformed Search Strategies

🔹 **BFS (Breadth-First Search)**  
✅ Expands level by level (shallowest nodes first).  
✅ **Complete** & **Optimal** (if step costs = 1).  
❌ **Memory-hungry** (stores all nodes at each level).
- **Time:** 𝑂(𝑏^𝑑)
- **Space:** 𝑂(𝑏^𝑑)


🔹 **UCS (Uniform Cost Search)**  
✅ Expands the **cheapest path** first.  
✅ **Optimal** but can be **slow** if small costs lead to deep searches.  
❌ Needs a **priority queue** → extra processing.
- **Time:** 𝑂(𝑏^(1 + ⌊C*/ε⌋)) (C* = optimal cost, ε = min step cost)
- **Space:** 𝑂(𝑏^(1 + ⌊C*/ε⌋))


🔹 **DFS (Depth-First Search)**  
✅ Dives **deep** before backtracking.  
✅ **Low memory use** but **not complete** (if infinite depth).  
❌ Can get **stuck in loops** or **miss shallow solutions**.
- **Time:** 𝑂(𝑏^𝑚)
- **Space:** 𝑂(𝑏𝑚)


🔹 **DLS (Depth-Limited Search)**  
✅ DFS but with a **depth limit** to avoid infinite loops.  
✅ **Better than DFS**, but **fails if the limit is too small**.
- **Time:** 𝑂(𝑏^𝑙) (l = depth limit)
- **Space:** 𝑂(𝑏𝑙)


🔹 **IDS (Iterative Deepening Search)**  
✅ Combines **BFS & DFS** → runs **DLS multiple times with increasing depth**.  
✅ **Complete, Optimal, & Saves Memory!**  
❌ **Repeats work**, but still efficient overall.
- **Time:** 𝑂(𝑏^𝑑)
- **Space:** 𝑂(𝑏𝑑)


🔹 **Bidirectional Search**  
✅ Searches **from start & goal** at the same time!  
✅ **Super fast** if both searches grow at the same rate.  
❌ **Hard to implement** (needs both searches to work well).
- **Time:** 𝑂(𝑏^(𝑑/2))
- **Space:** 𝑂(𝑏^(𝑑/2))


![[Pasted image 20250320014500.png]]



### **🚀 Informed (Heuristic) Search Strategies 🔥**

Unlike basic search, **heuristic search uses extra problem-specific knowledge** to make smarter moves.

- **f(n) = cost estimate**, picks the best node to expand first.
- **h(n) = heuristic function**, estimates the cheapest path from state **n** to the goal.
- **h(goal_state) = 0** (since reaching the goal has no extra cost).
- **Smarter expansion = Faster & more efficient search!**


🔹 **Greedy Best-First Search (GBFS)**  
✅ Uses **only** the heuristic (h(n)).  
✅ Fast but **not always optimal** (can get stuck in loops or bad paths).  
❌ Acts like an excited kid—always runs toward the closest goal-looking node but might get lost.
- **Time:** 𝑂(𝑏^𝑚) (worst case)
- **Space:** 𝑂(𝑏^𝑚)


🔹 __A* (Best of UCS + GBFS)__ 
✅ Uses **f(n) = g(n) + h(n)** (actual cost + estimated cost).  
✅ **Complete & Optimal** (if h(n) is admissible & consistent).  
❌ Uses **a lot of memory**!
- **Time:** 𝑂(𝑏^𝑑) (best case)
- **Space:** 𝑂(𝑏^𝑑) (holds all nodes in memory)


🔹 __IDA* (Iterative Deepening A*)__
✅ A* but with **less memory** → uses DFS-like cost limit.  
✅ **Complete & Optimal** but slower due to repetition.  
❌ Trades memory for **extra time**.
- **Time:** 𝑂(𝑏^𝑑)
- **Space:** 𝑂(𝑏𝑑)


🔹 **Weighted A****  
✅ A* but gives more importance to h(n): **f(n) = g(n) + w × h(n)** (w > 1).  
✅ **Faster but might not be optimal** (if w is too high).  
❌ More heuristic = More risky.
- **Time:** 𝑂(𝑏^𝑑)
- **Space:** 𝑂(𝑏^𝑑)


🔹 **RBFS (Recursive Best-First Search)**  
✅ A* but **uses recursion** to save memory.  
✅ **Less memory than A**** but **recomputes values**, making it slower.  
❌ Not practical for very large problems.
- **Time:** 𝑂(𝑏^𝑑)
- **Space:** 𝑂(𝑏𝑑)

🔹 __SMA* (Simplified Memory-Bounded A*)__  
✅ __A* with memory limits → keeps only best nodes.__ 
✅ **Balances memory & performance**, good for large problems.  
❌ Can **forget promising paths** if memory is too low.
- **Time:** 𝑂(𝑏^𝑑)
- **Space:** 𝑂(𝑚) (m = available memory)


### **🚀 Smarter Searching & Learning 🔥**

#### **Learning to Search Better**

An agent can **learn** to search smarter by understanding past searches.

- It explores a **metalevel state space** (thinking about its own thinking).
- The **search tree itself** is the internal state.
- Learning helps **avoid bad search paths**, cutting down cost and time.

#### **Heuristic Accuracy & Performance**

- Effective branching factor (b)* tells how well a heuristic guides the search.
- **Smaller b*** = **better heuristic** (close to 1 is ideal).
- If **h2(n) ≥ h1(n)**, then **h2 dominates h1** (expands fewer nodes).

#### **Generating Heuristics from Relaxed Problems**

- **Relaxed problem = fewer restrictions**, making solutions easier to find.
- Any **solution to the real problem** is also a solution to the relaxed one.
- The **best heuristic** is the most accurate one at each step.

#### **Learning Heuristics from Experience**

- Solve **many problems**, use their solutions to **train a heuristic**.
- Example: **8-puzzle**, learn h(n) from past solutions.
- Use **neural networks, decision trees, or inductive learning** to generalize better.

🔥 **End goal?** A search that **learns and improves** as it goes!