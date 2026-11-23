@compute @workgroup_size(64)
fn detectCollisions(
    @builtin(global_invocation_id) invocationId: vec3<u32>,
) {
    let i = invocationId.x;

    if (i >= params.bodyCount) {
        return;
    }

    let body = bodies[i];

    if (body.absorptionCooldown > 0u) {
        collisionCandidates[i] = i;
        return;
    }

    var candidate = i;
    var candidateMass = body.mass;

    for (var j = 0u; j < params.bodyCount; j++) {
        if (j == i) {
            continue;
        }

        let other = bodies[j];

        if (other.absorptionCooldown > 0u) {
            continue;
        }

        if (!hasHigherCollisionPriority(other.mass, j, candidateMass, candidate)) {
            continue;
        }

        if (spheresOverlap(body.position, body.radius, other.position, other.radius)) {
            candidate = j;
            candidateMass = other.mass;
        }
    }

    collisionCandidates[i] = candidate;
}