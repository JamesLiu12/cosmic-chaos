@group(0) @binding(4)
var<storage, read_write> donorCounts: array<atomic<u32>>;

@group(0) @binding(5)
var<storage, read> listOffsets: array<u32>;

@group(0) @binding(6)
var<storage, read_write> donorRanks: array<u32>;

@group(0) @binding(7)
var<storage, read_write> donorIndices: array<u32>;

@group(0) @binding(8)
var<storage, read_write> receiverAccepted: array<u32>;

fn isDirectDonor(donorIndex: u32, receiverIndex: u32) -> bool {
    return donorIndex != receiverIndex && collisionCandidates[receiverIndex] == receiverIndex;
}

@compute @workgroup_size(BODY_WORKGROUP_SIZE)
fn countDonors(@builtin(global_invocation_id) id: vec3<u32>) {
    let donorIndex = id.x;
    if (donorIndex >= params.bodyCount) {
        return;
    }

    let receiverIndex = collisionCandidates[donorIndex];
    if (isDirectDonor(donorIndex, receiverIndex)) {
        donorRanks[donorIndex] = atomicAdd(&donorCounts[receiverIndex], 1u);
    }
}

@compute @workgroup_size(BODY_WORKGROUP_SIZE)
fn scatterDonors(@builtin(global_invocation_id) id: vec3<u32>) {
    let donorIndex = id.x;
    if (donorIndex >= params.bodyCount) {
        return;
    }

    let receiverIndex = collisionCandidates[donorIndex];
    if (isDirectDonor(donorIndex, receiverIndex)) {
        let slot = listOffsets[receiverIndex] + donorRanks[donorIndex];
        donorIndices[slot] = donorIndex;
    }
}

@compute @workgroup_size(BODY_WORKGROUP_SIZE)
fn resolveReceivers(@builtin(global_invocation_id) id: vec3<u32>) {
    let receiverIndex = id.x;
    if (receiverIndex >= params.bodyCount || collisionCandidates[receiverIndex] != receiverIndex) {
        return;
    }

    let receiver = inputBodies[receiverIndex];
    let start = listOffsets[receiverIndex];
    let end = start + atomicLoad(&donorCounts[receiverIndex]);
    var gainedMass = 0.0;
    var radiusCubed = receiver.radius * receiver.radius * receiver.radius;
    for (var slot = start; slot < end; slot++) {
        let donor = inputBodies[donorIndices[slot]];
        let amount = transferredMass(donor, receiver);
        let donorRadiusCubed = donor.radius * donor.radius * donor.radius;
        gainedMass += amount;
        radiusCubed += donorRadiusCubed * (amount / donor.mass);
    }

    let transfer = resolveReceiver(receiver, gainedMass, radiusCubed);
    outputBodies[receiverIndex] = transfer.body;
    receiverAccepted[receiverIndex] = select(0u, 1u, transfer.accepted);
}

@compute @workgroup_size(BODY_WORKGROUP_SIZE)
fn updateDonors(@builtin(global_invocation_id) id: vec3<u32>) {
    let donorIndex = id.x;
    if (donorIndex >= params.bodyCount) {
        return;
    }

    let receiverIndex = collisionCandidates[donorIndex];
    if (receiverIndex == donorIndex) {
        return;
    }

    let donor = inputBodies[donorIndex];
    if (isDirectDonor(donorIndex, receiverIndex) && receiverAccepted[receiverIndex] != 0u) {
        outputBodies[donorIndex] = applyDonorTransfer(donor, inputBodies[receiverIndex]);
    } else {
        outputBodies[donorIndex] = donor;
    }
}
