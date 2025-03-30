possible investigations 
- uct score change to number of opponents / self pieces left? instead of -1, 1, 0 for win/loss 


# AlphaCheckers Development Roadmap

## Phase 1: Core Game Implementation
- [x] Implement basic Checkers rules
- [x] Create board initialization
- [x] Implement move validation
- [x] Add capture logic
- [x] Implement king promotion
- [x] Create game state management

## Phase 2: Basic MCTS AI Implementation
1. **Core MCTS Components**
   - [x] Complete _select() method in mcts.py
   - [x] Implement _expand() with random move selection
   - [x] Create _simulate() with random rollouts
   - [x] Add _backpropagate() for value updates

2. **AI Integration**
   - [x] Connect MCTS to main game loop
   - [x] Add basic AI vs human gameplay
   - [ ] Implement time-limited searches

## Phase 3: Neural Network Foundation
1. **Network Architecture**
   - [ ] Create network.py with basic CNN
   - [ ] Implement board state encoder
   - [ ] Add policy head for move probabilities
   - [ ] Add value head for position evaluation

2. **Training Infrastructure**
   - [ ] Create self_play.py
   - [ ] Implement data generation pipeline
   - [ ] Add experience replay buffer
   - [ ] Create basic training loop

## Phase 4: Integrated MCTS+NN
1. **Advanced MCTS**
   - [ ] Add neural network guidance to:
     - [ ] Expansion phase
     - [ ] Simulation phase
   - [ ] Implement prior probability integration
   - [ ] Add temperature-controlled exploration

2. **Optimization**
   - [ ] Implement parallel simulations
   - [ ] Add move pruning
   - [ ] Create caching system
   - [ ] Add virtual loss mechanism

## Phase 5: Evaluation & Deployment
1. **Training & Evaluation**
   - [ ] Create train.py
   - [ ] Implement policy gradient updates
   - [ ] Add Elo rating system
   - [ ] Create benchmark tests
   - [ ] Implement performance metrics

2. **User Experience**
   - [ ] Add difficulty levels
   - [ ] Implement save/load functionality
   - [ ] Create graphical interface
   - [ ] Add web interface (optional)
