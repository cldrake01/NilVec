pub const Metadata = union(enum) {
    u8: u8,
    u16: u16,
    u32: u32,
    u64: u64,
    usize: usize,

    i8: i8,
    i16: i16,
    i32: i32,
    i64: i64,
    isize: isize,

    f16: f16,
    f32: f32,
    f64: f64,

    bool: bool,

    str: []const u8,
};
