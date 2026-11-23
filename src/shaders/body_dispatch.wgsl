override BODY_WORKGROUP_SIZE: u32;

fn bodyWorkgroupCount(count: u32) -> u32 {
    return (count + BODY_WORKGROUP_SIZE - 1u) / BODY_WORKGROUP_SIZE;
}
