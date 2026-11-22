@compute @workgroup_size(64)
fn applyAbsorption(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.bodyCount) {
        return;
    }

    let body = inputBodies[i];

    let candidate = collisionCandidates[i];

    if (candidate != i) {
        if (collisionCandidates[candidate] != candidate) {
            outputBodies[i] = body;
        }

        return;
    }

    var gainedMass = 0.0;
    var radiusCubed = body.radius * body.radius * body.radius;

    for (var j = 0u; j < params.bodyCount; j++) {
        if (j == i || collisionCandidates[j] != i) {
            continue;
        }

        let donor = inputBodies[j];
        let amount = transferredMass(donor, body);
        let donorRadiusCubed = donor.radius * donor.radius * donor.radius;

        gainedMass += amount;
        radiusCubed += donorRadiusCubed * (amount / donor.mass);
    }

    let transfer = resolveReceiver(body, gainedMass, radiusCubed);
    outputBodies[i] = transfer.body;

    for (var j = 0u; j < params.bodyCount; j++) {
        if (j == i || collisionCandidates[j] != i) {
            continue;
        }

        let donor = inputBodies[j];
        if (transfer.accepted) {
            outputBodies[j] = applyDonorTransfer(donor, body);
        } else {
            outputBodies[j] = donor;
        }
    }
}