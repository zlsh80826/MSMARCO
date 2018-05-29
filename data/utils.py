import numpy as np

def smith_waterman(tt, bb):
    # adapted from https://gist.github.com/radaniba/11019717

    # These scores are taken from Wikipedia.
    # en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm
    match    = 2
    mismatch = -1
    gap      = -1

    def calc_score(matrix, x, y, seq1, seq2):
        '''Calculate score for a given x, y position in the scoring matrix.
        The score is based on the up, left, and upper-left neighbors.
        '''
        similarity = match if seq1[x - 1] == seq2[y - 1] else mismatch

        diag_score = matrix[x - 1, y - 1] + similarity
        up_score   = matrix[x - 1, y] + gap
        left_score = matrix[x, y - 1] + gap

        return max(0, diag_score, up_score, left_score)


    def create_score_matrix(rows, cols, seq1, seq2):
        '''Create a matrix of scores representing trial alignments of the two sequences.
        Sequence alignment can be treated as a graph search problem. This function
        creates a graph (2D matrix) of scores, which are based on trial alignments. 
        The path with the highest cummulative score is the best alignment.
        '''
        score_matrix = np.zeros((rows,cols))

        # Fill the scoring matrix.
        max_score = 0
        max_pos   = None    # The row and columbn of the highest score in matrix.
        for i in range(1, rows):
            for j in range(1, cols):
                score = calc_score(score_matrix, i, j, seq1, seq2)
                if score > max_score:
                    max_score = score
                    max_pos   = (i, j)

                score_matrix[i, j] = score

        if max_pos is None:
            raise ValueError('cannot align %s and %s'%(' '.join(seq1)[:80],' '.join(seq2)))

        return score_matrix, max_pos

    def next_move(score_matrix, x, y):
        diag = score_matrix[x - 1, y - 1]
        up   = score_matrix[x - 1, y]
        left = score_matrix[x, y - 1]
        if diag >= up and diag >= left:     # Tie goes to the DIAG move.
            return 1 if diag != 0 else 0    # 1 signals a DIAG move. 0 signals the end.
        elif up > diag and up >= left:      # Tie goes to UP move.
            return 2 if up != 0 else 0      # UP move or end.
        elif left > diag and left > up:
            return 3 if left != 0 else 0    # LEFT move or end.
        else:
            # Execution should not reach here.
            print('qq')
            raise ValueError('invalid move during traceback')

    def traceback(score_matrix, start_pos, seq1, seq2):
        '''Find the optimal path through the matrix.
        This function traces a path from the bottom-right to the top-left corner of
        the scoring matrix. Each move corresponds to a match, mismatch, or gap in one
        or both of the sequences being aligned. Moves are determined by the score of
        three adjacent squares: the upper square, the left square, and the diagonal
        upper-left square.
        WHAT EACH MOVE REPRESENTS
            diagonal: match/mismatch
            up:       gap in sequence 1
            left:     gap in sequence 2
        '''

        END, DIAG, UP, LEFT = range(4)
        x, y         = start_pos
        move         = next_move(score_matrix, x, y)
        while move != END:
            if move == DIAG:
                x -= 1
                y -= 1
            elif move == UP:
                x -= 1
            else:
                y -= 1
            move = next_move(score_matrix, x, y)

        return (x,y), start_pos

    rows = len(tt) + 1
    cols = len(bb) + 1

    # Initialize the scoring matrix.
    score_matrix, start_pos = create_score_matrix(rows, cols, tt, bb)
    
    # Traceback. Find the optimal path through the scoring matrix. This path
    # corresponds to the optimal local sequence alignment.
    (x,y), (w,z) = traceback(score_matrix, start_pos, tt, bb)
    return (x,w), (y,z), score_matrix[w][z]