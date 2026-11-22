@group(0) @binding(0)
var<storage, read> inputBodies: array<Body>;

@group(0) @binding(1)
var<storage, read_write> outputBodies: array<Body>;

@group(0) @binding(2)
var<storage, read> collisionCandidates: array<u32>;

@group(0) @binding(3)
var<uniform> params: SimulationUniforms;

const MASS_PER_DISTANCE: f32 = 0.5;
const MIN_ABSORPTION_RATE: f32 = 0.5;
const MAX_ABSORPTION_RATE: f32 = 4.0;

fn transferredMass(donor: Body, receiver: Body) -> f32 {
    let relativeSpeed = length(donor.velocity - receiver.velocity);
    let rate = clamp(
        MASS_PER_DISTANCE * relativeSpeed,
        MIN_ABSORPTION_RATE,
        MAX_ABSORPTION_RATE
    );

    return min(donor.mass, rate * params.fixedDt);
}

struct ReceiverTransfer {
    body: Body,
    accepted: bool,
}

fn resolveReceiver(body: Body, gainedMass: f32, radiusCubed: f32) -> ReceiverTransfer {
    var receiver = body;
    var acceptsTransfer = gainedMass > 0.0;

    if (acceptsTransfer) {
        receiver.mass = body.mass + gainedMass;
        receiver.radius = pow(radiusCubed, 1.0 / 3.0);

        if (params.boundaryEnabled != 0u) {
            let radius = vec3<f32>(receiver.radius);
            let lo = params.boundaryMin + radius;
            let hi = params.boundaryMax - radius;

            if (any(lo >= hi)) {
                acceptsTransfer = false;
                receiver = body;
            } else {
                receiver.position = clamp(receiver.position, lo, hi);

                for (var axis = 0u; axis < 3u; axis++) {
                    if (receiver.position[axis] == lo[axis]) {
                        receiver.velocity[axis] = abs(receiver.velocity[axis]);
                    } else if (receiver.position[axis] == hi[axis]) {
                        receiver.velocity[axis] = -abs(receiver.velocity[axis]);
                    }
                }
            }
        }
    }

    return ReceiverTransfer(receiver, acceptsTransfer);
}

fn applyDonorTransfer(donor: Body, receiver: Body) -> Body {
    let amount = transferredMass(donor, receiver);
    if (amount == donor.mass) {
        return Body();
    }

    var result = donor;
    if (amount > 0.0) {
        result.mass = donor.mass - amount;
        let remainingFraction = result.mass / donor.mass;
        result.radius = donor.radius * pow(remainingFraction, 1.0 / 3.0);
    }
    return result;
}