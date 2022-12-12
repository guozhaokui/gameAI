import { DQNAgent } from "./DQNAgent";

interface IEnv {
    getNumStates(): number;
    getMaxNumActions(): number;
}

//
export function testRL() {
    // create an environment object
    var env = {
        getNumStates() { return 8; },
        getMaxNumActions() { return 4; }
    };

    // create the DQN agent
    var spec = { alpha: 0.01 } // see full options on DQN page
    let agent = new DQNAgent(env, spec);

    setInterval(function () { // start the learning loop
        var action = agent.act([1,0,0,0,0,0,0,0]); // s is an array of length 8
        //... execute action in environment and get the reward
        let reward = 1;
        agent.learn(reward); // the agent improves its Q,policy,model, etc. reward is a float
    }, 0);
}