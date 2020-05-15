import sys
import copy
import numpy as np
from queue import PriorityQueue

PUZZLE_WIDTH = 4
BLANK = 0  # Integer comparison tends to be faster than string comparison
BETTER = True

# Read a NumberPuzzle from stdin; space-delimited, blank is "-"
def read_puzzle():
    new_puzzle = NumberPuzzle()
    row = 0
    for line in sys.stdin:
        tokens = line.split()
        for i in range(PUZZLE_WIDTH):
            if tokens[i] == '-':
                new_puzzle.tiles[row][i] = BLANK
                new_puzzle.blank_r = row
                new_puzzle.blank_c = i
            else:
                try:
                    new_puzzle.tiles[row][i] = int(tokens[i])
                except ValueError:
                    sys.exit("Found unexpected non-integer for tile value")
        row += 1
    return new_puzzle

# Class containing an int array for tiles
# blank_c and blank_r for column and row of the blank
# (so we don't have to hunt for it)
class NumberPuzzle:
    # This is a constructor in Python - just return zeros for everything
    # and fill in the tile array later
    def __init__(self):
        self.tiles = np.zeros((PUZZLE_WIDTH,PUZZLE_WIDTH))
        self.blank_r = 0
        self.blank_c = 0
        # This next field is for our convenience when generating a solution
        # -- remember which puzzle was the move before
        self.parent = None
        self.dist_from_start = 0
        self.key = 0

    # This is the Python equivalent of Java's toString()
    def __str__(self):
        out = ""
        for i in range(PUZZLE_WIDTH):
            for j in range(PUZZLE_WIDTH):
                if j > 0:
                    out += " "
                if self.tiles[i][j] == BLANK:
                    out += "-"
                else:
                    out += str(int(self.tiles[i][j]))
            out += "\n"
        return out

    # In A* search, we generally want to copy instead of destructively alter,
    # since we're not backtracking so much as jumping around the search tree.
    # Also, if A and B are numpy arrays, "A = B" only passes a reference to B.
    # "deepcopy" copies out the data (recursively if need be, a fact possibly
    # useful for other assignments).
        # We'll also use this to tell the child we're its parent
    def copy(self):
        child = NumberPuzzle()
        child.tiles = np.copy(self.tiles)
        child.blank_r = self.blank_r
        child.blank_c = self.blank_c
        child.dist_from_start = self.dist_from_start
        child.parent = self
        return child

    # Overrides == for this object so that we can compare by tile arrangement
    # instead of reference.  This is going to be pretty common, so we'll skip
    # a type check on "other" for a modest speed increase
    def __eq__(self, other):
        return np.array_equal(self.tiles, other.tiles)

    # Hash function necessary for inclusion in a set -- unique "name"
    # for this object -- we'll just hash the bytes of the tile array
    def __hash__(self):
        return hash(bytes(self.tiles))

    # Override less-than so that we can put these in a priority queue
    # with no problem.  We don't want to recompute the heuristic here, 
    # though -- that would be too slow to do it every time we need to
    # reorganize the priority queue 

    def __lt__(self, obj):
        return (self.key < obj.key) or (self.key == obj.key and self.dist_from_start > obj.dist_from_start)

    # Move from the row, column coordinates given into the blank.
    # Also very common, so we will also skip checks for legality to improve
    # speed.
    def move(self, tile_row, tile_column):
        self.tiles[self.blank_r][self.blank_c] = self.tiles[tile_row][tile_column]
        self.tiles[tile_row][tile_column] = BLANK
        self.blank_r = tile_row
        self.blank_c = tile_column
        self.dist_from_start += 1

    # Return a list of NumberPuzzle states that could result from one
    # move on the present board.  Use this to keep the order in which
    # moves are evaluated the same as our solution, thus matching the
    # HackerRank solution as well.  (Also notice we're still in the
    # methods of NumberPuzzle, hence the lack of arguments.)
    def legal_moves(self):
        legal = []
        if (self.blank_r > 0):
            down_result = self.copy()
            down_result.move(self.blank_r-1, self.blank_c)
            legal.append(down_result)
        if (self.blank_c > 0):
            right_result = self.copy()
            right_result.move(self.blank_r, self.blank_c-1)
            legal.append(right_result)
        if (self.blank_r < PUZZLE_WIDTH - 1):
            up_result = self.copy()
            up_result.move(self.blank_r+1, self.blank_c)
            legal.append(up_result)
        if (self.blank_c < PUZZLE_WIDTH - 1):
            left_result = self.copy()
            left_result.move(self.blank_r, self.blank_c+1)
            legal.append(left_result)
        return legal
        
    # solve: return a list of puzzle states from this state to solved
    # better_h flag determines whether to use the better heuristic
    def solve(self, better_h):
        #heuristic flag
        self.key = self.heuristic(better_h) + self.dist_from_start
        # TODO
        #initialize priority queue; each node = cost to get there + estimated cost to go
        queue = PriorityQueue()
        queue.put(self.copy())
        #intitalize table of distances
        #initialize set with already seen, possibly add current to that set
        #initialize table of distances to each node
        seen = set()
        #seen.add(queue[0].)
        while queue.qsize() > 0:
            n = queue.get()
            #remove lowest-cost node N
            if n.solved():
                #return path to n
                return n.find_path()
            #if n in seen or we know as-good distance -> continue
            if n in seen:
                continue
            for state in seen:
                if n.__lt__(state):
                    continue
            #record cost to get here in table of distances
            #For each neighbor of this node (all possible moves)
                #Add to priority queue with appropriate cost to get there + heuristic
                #Remember N is our parent
            seen.add(n)
            for state in n.legal_moves():
                s = state.copy()
                s.dist_from_start += 1
                s.parent = n
                s.key = s.heuristic(better_h) + s.dist_from_start
                queue.put(s)
        #return no path (if priority queue is empty w/o path to goal)
        return None


    def find_path(self):
        path = []
        n = self.copy()
        while n.parent:
            path.append(n)
            n = n.parent
        return path

    # solved(): return True iff all tiles in order and blank in bottom right
    def solved(self):
        should_be = 1
        for i in range(PUZZLE_WIDTH):
            for j in range(PUZZLE_WIDTH):
                if self.tiles[i][j] != should_be:
                    return False
                else:
                    should_be = (should_be + 1) % (PUZZLE_WIDTH ** 2)
        return True

    # Toggle which heuristic is used
    def heuristic(self, better_h):
        if better_h:
            return self.manhattan_heuristic()
        return self.tile_mismatch_heuristic()

    # Count tiles out of place.
    def tile_mismatch_heuristic(self):
        mismatch_count = 0
        should_be = 1
        for i in range(PUZZLE_WIDTH):
            for j in range(PUZZLE_WIDTH):
                if self.tiles[i][j] != should_be:
                    mismatch_count+=1
                should_be = (should_be + 1) % (PUZZLE_WIDTH ** 2)
        return mismatch_count

    # Returns total manhattan (city block) distance needed for all tiles.  
    def manhattan_heuristic(self):
        total_manhattan = 0
        for i in range(PUZZLE_WIDTH):
            for j in range(PUZZLE_WIDTH):
                if self.tiles[i][j] != '-':
                    x, y = divmod(self.tiles[i][j]-1, PUZZLE_WIDTH)
                    total_manhattan += abs(x - i) + abs(y - j)
        return total_manhattan

# Print every puzzle in the list
def print_steps(path):
    if path is None:
        print("No path found")
    else:
        for state in path:
            print(state)

# main
my_puzzle = read_puzzle()
solution_steps = my_puzzle.solve(BETTER)
print_steps(solution_steps)
