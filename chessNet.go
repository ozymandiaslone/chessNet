package chessNet

import (
    "fmt"
    "math/rand"
	"github.com/notnil/chess"
)

const populationSize = 1000 // number of neural networks in the population

type NeuralNetwork struct {
	// Field representing the quantity of input neurons
    numInputs int
	// Field representing the quantity of output neurons
    numOutputs int
	// Field representing the quantity of hidden layers
    numHiddenLayers int
	// Field representing the quantity of neurons per hidden layer
    numNeuronsPerHiddenLayer int

	// A three-dimensional slice we will use to store the network's weights
	// The first dimension corresponds to the layer within the network
	// The second dimension corresponds to the neuron within that layer (not including output layer)
	// The third dimension corresponds to the neuron which the second-dimension-neuron is connected to
    weights [][][]float64

	// Biases are stored in a two dimensional slice, because the bias information is stored in each individual neuron
	// First dimension corresonds to the layer (including output layer)
	// Second dimension corresponds to the neuron within the layer
    biases [][]float64
}


// Initialize the weights and biases of the neural network using random values
func (nn *NeuralNetwork) Initialize(inputs, outputs, hiddenLayers, neuronsPerLayer int) {

	// Set the size of the network 
	nn.numInputs = inputs
	nn.numOutputs = outputs
	nn.numHiddenLayers = hiddenLayers
	nn.numNeuronsPerHiddenLayer = neuronsPerLayer

    // Initialize the weights and biases using small random values
    nn.weights = make([][][]float64, nn.numHiddenLayers+1)
    nn.biases = make([][]float64, nn.numHiddenLayers+2)
    for i := 0; i < nn.numHiddenLayers+1; i++ {
        nn.weights[i] = make([][]float64, nn.numNeuronsPerHiddenLayer)
        nn.biases[i] = make([]float64, nn.numNeuronsPerHiddenLayer)
        for j := 0; j < nn.numNeuronsPerHiddenLayer; j++ {
            if i == 0 {
                nn.weights[i][j] = make([]float64, nn.numInputs)
            } else if i == nn.numHiddenLayers {
                nn.weights[i][j] = make([]float64, nn.numOutputs)
            } else {
                nn.weights[i][j] = make([]float64, nn.numNeuronsPerHiddenLayer)
            }
            for k := range nn.weights[i][j] {
                nn.weights[i][j][k] = rand.Float64()
            }
            nn.biases[i][j] = rand.Float64()
        }
    }
}



// Function to breed two neural networks with no genetic loss
func perfectBreed(parent1, parent2 *NeuralNetwork) *NeuralNetwork{
	// Set the size to be the same as the first parent (all sizes should be uniform right now)
	child := &NeuralNetwork{}
	child.numInputs = parent1.numInputs
	child.numOutputs = parent1.numOutputs
	child.numHiddenLayers = parent1.numHiddenLayers
	child.numNeuronsPerHiddenLayer = parent1.numNeuronsPerHiddenLayer

	// Create three dimensional and two dimensional slices for the weights and biases of the network, setting the size of the first dimension approproately
	child.weights = make([][][]float64, child.numHiddenLayers + 1)
	child.biases = make([][]float64, child.numHiddenLayers + 2)

	for i := 0; i < child.numHiddenLayers + 1; i++ {
		child.weights[i] = make([][]float64, child.numNeuronsPerHiddenLayer)
        for j := 0; j< child.numNeuronsPerHiddenLayer; j++ {
            // Check if we're working with the input layer
            if i == 0 {
                child.weights[i][j] = make([]float64, child.numInputs)
            }
            // Check if we are working with the output layer
            else if i == child.numHiddenLayers {
                child.weights[i][j] = make([]float64, child.numOutputs) 
            // Otherwise, its a hidden layer
            } else {
                child.weights[i][j] = make([]float64, child.numNeuronsPerHiddenLayer)
            }
        }
	}
	for i := 0; i < child.numHiddenLayers + 2; i++ {
		child.biases[i] = make([]float64, child.numNeuronsPerHiddenLayer)
	}
    // chromosome type crossover (via borderline black magic at some points)
    rand.Seed(time.Now().UTC().UnixNano())

    // Generate four random x values with no repeats
            //(oooooo maaathhh)
    xValues := rand.Perm(child.numHiddenLayers + 1)[:4]
    sort.Ints(xValues)

    // Generate a random y value
    // We are about to literally play God. Imagine the lines we're generating for our net swaps. 
    // Lets keep these lines from having too crazy of an angle, relative to the range of the coordinates we have.
    yValue := rand.Intn((child.numNeuronsPerHiddenLayer /5) + 1)

    // Calculate the slope of the first line
    slope := float64(yValue) / float64(xValues[1] - xValues[0])

    // Calculate the y-intercept of the first line
    yIntercept := yValue - int(slope * float64(xValues[0]))

    // Calculate the y-values for the remaining lines using the slope and x-values
    yValues := make([]int, 3)
    yValues[0] = yIntercept
    for i := 1; i < 3; i++ {
        yValues[i] = int(slope * float64(xValues[i + 1])) + yIntercept
    }
    // Set the points using the calculated x and y values
    p1 := xValues[0]
    p2 := yValues[0]
    p3 := xValues[1]
    p4 := yValues[1]
    p5 := xValues[2]
    p6 := yValues[2]
    p7 := xValues[3]
    p8 := yValues[3]


    // CROSSOVER TIME WOOOO

    // Iterate through neruons...
    for i := 0; i < child.numHiddenLayers + 1; i++ {
        for j := 0; j < child.numNeuronsPerHiddenLayer; j++ {

            // Check which line the point (i, j) falls under
            if i < p1 || (i == p1 && j < p2) {

                // Point falls under line 1, copy from parent 1
                child.weights[i][j] = parent1.weights[i][j]
                child.biases[i] = parent1.biases[i]

            } else if (i > p1 && i < p3) || (i == p3 && j < p4) {

                // Point falls under line 2, copy from parent 2
                child.weights[i][j] = parent2.weights[i][j]
                child.biases[i] = parent2.biases[i]

            } else if (i > p3 && i < p5) || (i == p5 && j < p6) {

                // Point falls under line 3, copy from parent 1
                child.weights[i][j] = parent1.weights[i][j]
                child.biases[i] = parent1.biases[i]

            } else if (i > p5 && i < p7) || (i == p7 && j < p8) {

                // Point falls under line 4, copy from parent 2
                child.weights[i][j] = parent2.weights[i][j]
                child.biases[i] = parent2.biases[i]
            } else {

                // Point falls outside all lines, copy from parent 1

                child.weights[i][j] = parent1.weights[i][j]
                child.biases[i] = parent1.biases[i]
            }
        }
    }

    return child 

}

// Function to breed two neural networks with slight genetic mutation
func imperfectBreed(parent1, parent2 *NeuralNetwork) *NeuralNetwork {
	// Set the size to be the same as the first parent (all sizes should be uniform right now)
	child := &NeuralNetwork{}
	child.numInputs = parent1.numInputs
	child.numOutputs = parent1.numOutputs
	child.numHiddenLayers = parent1.numHiddenLayers
	child.numNeuronsPerHiddenLayer = parent1.numNeuronsPerHiddenLayer

	// Create three dimensional and two dimensional slices for the weights and biases of the network, setting the size of the first dimension approproately
	child.weights = make([][][]float64, child.numHiddenLayers + 1)
	child.biases = make([][]float64, child.numHiddenLayers + 2)

	for i := 0; i < child.numHiddenLayers + 1; i++ {
		child.weights[i] = make([][]float64, child.numNeuronsPerHiddenLayer)
        for j := 0; j< child.numNeuronsPerHiddenLayer; j++ {
            // Check if we're working with the input layer
            if i == 0 {
                child.weights[i][j] = make([]float64, child.numInputs)
            }
            // Check if we are working with the output layer
            else if i == child.numHiddenLayers {
                child.weights[i][j] = make([]float64, child.numOutputs) 
            // Otherwise, its a hidden layer
            } else {
                child.weights[i][j] = make([]float64, child.numNeuronsPerHiddenLayer)
            }
        }
	} else {
		for i := 0; i < child.numHiddenLayers + 2; i++ {
			child.biases[i] = make([]float64, child.numNeuronsPerHiddenLayer)
		}
		// chromosome type crossover (via borderline black magic at some points)
		rand.Seed(time.Now().UTC().UnixNano())

		// Generate four random x values with no repeats
				//(oooooo maaathhh)
		xValues := rand.Perm(child.numHiddenLayers + 1)[:4]
		sort.Ints(xValues)

		// Generate a random y value
		// We are about to literally play God. Imagine the lines we're generating for our net swaps. 
		// Lets keep these lines from having too crazy of an angle, relative to the range of the coordinates we have.
		yValue := rand.Intn((child.numNeuronsPerHiddenLayer /5) + 1)

		// Calculate the slope of the first line
		slope := float64(yValue) / float64(xValues[1] - xValues[0])

		// Calculate the y-intercept of the first line
		yIntercept := yValue - int(slope * float64(xValues[0]))

		// Calculate the y-values for the remaining lines using the slope and x-values
		yValues := make([]int, 3)
		yValues[0] = yIntercept
		for i := 1; i < 3; i++ {
			yValues[i] = int(slope * float64(xValues[i + 1])) + yIntercept
		}
		// Set the points using the calculated x and y values
		p1 := xValues[0]
		p2 := yValues[0]
		p3 := xValues[1]
		p4 := yValues[1]
		p5 := xValues[2]
		p6 := yValues[2]
		p7 := xValues[3]
		p8 := yValues[3]


		// CROSSOVER TIME WOOOO

		// Iterate through neruons...
		for i := 0; i < child.numHiddenLayers + 1; i++ {
			for j := 0; j < child.numNeuronsPerHiddenLayer; j++ {

				// Decide whether to mutate this neuron 
				if rand.Intn(166) == 0 {
					// If so, randomize
					for k := range child.weights[i][j] {
						child.weights[i][k][j] = rand.Float64()
					}
				}

				// Check which line the point (i, j) falls under
				if i < p1 || (i == p1 && j < p2) {

					// Point falls under line 1, copy from parent 1
					child.weights[i][j] = parent1.weights[i][j]
					child.biases[i] = parent1.biases[i]

				} else if (i > p1 && i < p3) || (i == p3 && j < p4) {

					// Point falls under line 2, copy from parent 2
					child.weights[i][j] = parent2.weights[i][j]
					child.biases[i] = parent2.biases[i]

				} else if (i > p3 && i < p5) || (i == p5 && j < p6) {

					// Point falls under line 3, copy from parent 1
					child.weights[i][j] = parent1.weights[i][j]
					child.biases[i] = parent1.biases[i]

				} else if (i > p5 && i < p7) || (i == p7 && j < p8) {

					// Point falls under line 4, copy from parent 2
					child.weights[i][j] = parent2.weights[i][j]
					child.biases[i] = parent2.biases[i]
				} else {

					// Point falls outside all lines, copy from parent 1

					child.weights[i][j] = parent1.weights[i][j]
					child.biases[i] = parent1.biases[i]
				}
			}
		}
	}

    return child 

}

func playGame(nn1, nn2 *NeuralNetwork) int {
	// TODO
	// Figure out how tf this parts gonna work
	// Initialize chess board
	game := chess.NewGame()

	moves := game.ValidMoves()
	fmt.Println(moves)
	// this returns the following chess format: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
	fmt.Println(game.Position())
	
}

func hotEncode(fen string) [][]int {
	// Split the fen into its respective fields
	fenFields := strings.Split(fen, " ")

	// create 2 dimensional slice to represent the hot encoded board info
	board := make([][]int, 64)
	for i := 0; i < 64; i++ {
		board[i] = make([], 7)
	}
	
	// stupid boring fen math
	pieces := strings.Split(fenFields[0], "/")
	// Remember there are eight rows in chess
	for i := 0; i < 8; i++ {
		for j := 0; j < len(fenFields[i]); j++ {
			// calculate the index of the current sqare we're working with
			id := i*8 + j

			// Figure out the piece 
			piece := pieces[i][j]
			// Encode that shit
			switch piece {
			case "P":
				board[id][1] = 1
			case "N":
				board[id][2] = 1
			case "B":
				board[id][3] = 1
			case "R":
				board[id][4] = 1
			case "Q":
				board[id][5] = 1
			case "K":
				board[id][6] = 1
			case "p":
				board[id][0] = 1
				board[id][1] = 1
			case "n":
				board[id][2] = 1
				board[id][0] = 1
			case "b":
				board[id][3] = 1
				board[id][0] = 1
			case "r":
				board[id][4] = 1
				board[id][0] = 1
			case "q":
				board[id][5] = 1
				board[id][0] = 1
			case "k":
				board[id][6] = 1
				board[id][0] = 1
			default:
				// If it aint a piece letter, it's gotta be a number for the empty squares
				num, _ := strconv.Atoi(string(piece))
				id += num - 1
			}
		}
	}
	// Now we add the turn. We gotta make a special 2d thingy to tack it on the end
	turn := 0
	if fenFields[1] == "b" {
		turn == 1
	}
	board = append([]int{turn}, board...)\

	return board
}

