use std::cell::Ref;
use std::collections::HashMap;
use std::rc::Rc;

use ga::{self, Array, Tensor, TensorMode};
use rand;

use super::init::Initializer;
use super::op::{OpBuilder, OpDescriptor, Operation};
use super::var_store::{VarIndex, VarStore};

#[derive(Copy, Clone)]
pub enum NodeInput {
    Var(VarIndex),    // Regular input variable
    Recurrent(usize), // Recurrent connection
}

pub struct Node {
    pub inputs: Vec<VarIndex>,
    pub outputs: Vec<VarIndex>,
    pub in_grad: Vec<VarIndex>, // gradients on inputs
    pub out_grad: Vec<OutGrad>, // gradients on outputs
}

pub struct Graph {
    ctx: Rc<ga::Context>,
    nodes: Vec<Node>,
    node_ops: Vec<Box<Operation>>,
    pub var_store: VarStore,
    out_var_map: HashMap<VarIndex, (NodeIndex, usize)>, // Maps output variable to its node and index within node
    // Gradients on variables that are inputs to the graph - they have no corresponding node
    in_var_map: HashMap<VarIndex, usize>,
    learnables: Vec<(VarIndex, GradIndex)>, // Learnable variables
    in_grad: Vec<OutGrad>, // Gradients on variables that are inputs to the graph

    rng: rand::ThreadRng,
}

impl Graph {
    pub fn new(ctx: Rc<ga::Context>) -> Self {
        Graph {
            ctx: ctx,
            nodes: vec![],
            node_ops: vec![],
            var_store: VarStore::new(),
            out_var_map: HashMap::new(),
            in_var_map: HashMap::new(),
            learnables: vec![],
            in_grad: vec![],
            rng: rand::thread_rng(),
        }
    }

    pub fn add_node<T: OpBuilder>(&mut self, op: T) -> NodeIndex {
        let node_index = NodeIndex(self.nodes.len());

        let OpDescriptor { op, inputs: node_inputs, out_shapes } = op.build(&self.ctx, &self.var_store).unwrap();

        // Create output variables
        let mut outputs = vec![];
        for (i, shape) in out_shapes.into_iter().enumerate() {
            let var_index = self.var_store.add(Tensor::new(self.ctx.as_ref(), shape, TensorMode::Mut));
            outputs.push(var_index);
            self.out_var_map.insert(var_index, (node_index, i));
        }
        let mut out_grad = vec![OutGrad::new(); outputs.len()];
        // Set up inputs and gradients on inputs
        let mut inputs = vec![];
        let mut in_grad = vec![];
        for input in node_inputs {
            let (v, gradient) =
                match input {
                    NodeInput::Var(v) => (v, self.add_gradient(v)),
                    NodeInput::Recurrent(out) => {
                        let v = outputs[out];
                        let gradient = Self::create_gradient(&self.ctx, &mut self.var_store, v,
                                                             &mut out_grad[out]);
                        (v, gradient)
                    }
                };
            inputs.push(v);
            in_grad.push(gradient);
        }
        // Create the node
        self.nodes.push(Node { inputs: inputs,
                               outputs: outputs,
                               in_grad: in_grad,
                               out_grad: out_grad });
        // Add the corresponding node op
        self.node_ops.push(Box::new(op));
        node_index
    }

    pub fn add_variable<I: Initializer>(&mut self,
                                        shape: Vec<usize>,
                                        learnable: bool,
                                        init: I) -> VarIndex {
        let a = init.init(&mut self.rng, shape);
        let v = self.var_store.add(Tensor::from_array(&self.ctx, &a, TensorMode::Mut));
        self.in_var_map.insert(v, self.in_grad.len());
        if learnable {
            self.learnables.push((v, GradIndex::InVar(self.in_grad.len())));
        }
        self.in_grad.push(OutGrad::new());
        v
    }

    pub fn get_gradient(&self, v: VarIndex) -> GradIndex {
        match self.out_var_map.get(&v).map(|x| *x) {
            Some((node, out_index)) => {
                // v is the output of a node
                GradIndex::OutVar(node, out_index)
            },
            None => {
                // v is an input to the graph - it has no corresponding node
                let in_grad_index = *self.in_var_map.get(&v)
                                         .expect("Variable is neither input nor output. Nonsense!");
                GradIndex::InVar(in_grad_index)
            },
        }
    }

    pub fn add_gradient(&mut self, v: VarIndex) -> VarIndex {
        match self.out_var_map.get(&v).map(|x| *x) {
            Some((node, out_index)) => {
                // v is the output of a node
                /*self.nodes[node.0].out_grad[out_index]
                                  .fork(&self.ctx, &mut self.var_store, gradient);*/
                Self::create_gradient(&self.ctx, &mut self.var_store, v,
                                      &mut self.nodes[node.0].out_grad[out_index])
            },
            None => {
                // v is an input to the graph - it has no corresponding node
                let in_grad_index = *self.in_var_map.get(&v)
                                         .expect("Variable is neither input nor output. Nonsense!");
                /*self.in_grad[in_grad_index]
                    .fork(&self.ctx, &mut self.var_store, gradient);*/
                Self::create_gradient(&self.ctx, &mut self.var_store, v,
                                      &mut self.in_grad[in_grad_index])
            },
        }
    }

    fn create_gradient(ctx: &ga::Context,
                       var_store: &mut VarStore,
                       v: VarIndex,
                       out_grad: &mut OutGrad)
                       -> VarIndex {
        let shape = var_store.get(v).shape().to_owned();
        let gradient = var_store.add(Tensor::new(ctx, shape, TensorMode::Mut));
        out_grad.fork(ctx, var_store, gradient);
        gradient
    }

    pub fn forward(&mut self) {
        // Forward pass
        //
        // NOTE: We just execute the nodes in order. We can do this because of the way the graph is
        // built. When a user wants to add a node, he/she must also supply the inputs. This means
        // any dependencies must already be added before the node can be added. Therefore, we can
        // assert that all dependents come after their dependencies in the `self.nodes` array.
        for (node, op) in self.nodes.iter().zip(&mut self.node_ops) {
            op.forward(&self.ctx, &self.var_store, node);
        }
    }

    pub fn backward(&mut self) {
        // Backward pass
        // We run through the nodes in reverse order. See note in Graph::forward
        for (node, op) in self.nodes.iter_mut().rev().zip(self.node_ops.iter_mut().rev()) {
            // Sum the gradients on each output if there are multiple gradients
            for out_grad in &node.out_grad {
                out_grad.maybe_sum(self.ctx.as_ref(), &mut self.var_store);
            }
            op.backward(&self.ctx, &mut self.var_store, node);
        }
    }

    pub fn context(&self) -> &ga::Context {
        &self.ctx
    }

    pub fn learnables(&self) -> &[(VarIndex, GradIndex)] {
        &self.learnables
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct OutGrad {
    gradient: Option<VarIndex>, // The gradient or sum of gradients
    gradients: Vec<VarIndex>,
}

impl OutGrad {
    pub fn new() -> Self {
        OutGrad {
            gradient: None,
            gradients: vec![],
        }
    }

    pub fn get(&self) -> VarIndex {
        self.gradient.unwrap()
    }

    pub fn try_get(&self) -> Option<VarIndex> {
        self.gradient
    }

    fn maybe_sum(&self, ctx: &ga::Context, var_store: &mut VarStore) {
        if self.gradients.len() > 0 {
            if let Some(sum) = self.gradient {
                ga::copy_to(ctx, &var_store.get(self.gradients[0]), &var_store.get(sum));
                for grad in &self.gradients[1..] {
                    ga::add(ctx, &var_store.get(sum), -1, &var_store.get(*grad), &var_store.get(sum));
                }
            }
        }
    }

    fn fork(&mut self, ctx: &ga::Context, var_store: &mut VarStore, v: VarIndex) {
        if self.gradients.len() > 0 {
            // There are multiple gradients already, just add the new one to the list
            self.gradients.push(v);
        } else if let Some(gradient) = self.gradient {
            // There is still only one gradient, switch it to a fork
            let shape = {
                let grad = var_store.get(gradient);
                grad.shape().to_vec()
            };
            // Create variable for gradient sum
            self.gradient = Some(var_store.add(Tensor::new(ctx, shape, TensorMode::Mut)));
            self.gradients.push(gradient);
            self.gradients.push(v);
        } else {
            // This is the only gradient so far, so we don't need to sum anything
            self.gradient = Some(v);
        }
    }
}

#[derive(Copy, Clone)]
pub enum GradIndex {
    InVar(usize),
    OutVar(NodeIndex, usize),
}

impl GradIndex {
    pub fn get<'a>(&self, graph: &'a Graph) -> Ref<'a, Tensor<f32>> {
        match *self {
            GradIndex::InVar(in_grad_index) => {
                graph.in_grad[in_grad_index].get().get(graph)
            },
            GradIndex::OutVar(node, out_index) => {
                node.get(graph).out_grad[out_index].get().get(graph)
            },
        }
    }

    pub fn read(&self, g: &Graph, a: &mut Array<f32>) {
        self.get(g).read(g.context(), a);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone)]
pub struct NodeIndex(usize);

impl NodeIndex {
    pub fn get<'a>(&self, g: &'a Graph) -> &'a Node {
        &g.nodes[self.0]
    }
}

#[test]
fn it_works() {
    use super::op::MatMul;
    
    let ctx = Rc::new(ga::Context::new());

    // Setup the graph
    let mut graph = Graph::new(ctx.clone());
    let a = graph.add_variable(vec![1, 2], true, vec![1.4, 0.3]);
    let wa = graph.add_variable(vec![2, 3], true, vec![0.5, 0.3, 0.2,
                                                       0.6, 0.7, 0.7]);
    let node = graph.add_node(MatMul(a, wa));
    let node_out = node.get(&graph).outputs[0];
    let node_g = graph.add_gradient(node_out);

    // Send some input data
    let node_g_cpu = ga::Array::from_vec(vec![1, 3], vec![1.0, -1.0, 0.5]);
    node_g.get(&graph).set(&ctx, &node_g_cpu);

    // Run the network
    graph.forward();
    graph.backward();
    let out = node.get(&graph).outputs[0].get(&graph).get(&ctx);
    let wa_d = graph.get_gradient(wa).get(&graph).get(&ctx);
    println!("out = {:?}", out);
    println!("wa_d = {:?}", wa_d);
    assert!(out.buffer() == &[0.88, 0.63, 0.49]);
    assert!(wa_d.buffer() == &[1.4, -1.4, 0.7,
                               0.3, -0.3, 0.15]);
}
