const std = @import("std");
const ID = @import("ID.zig").ID;

pub const Candidate = struct {
    const Self = @This();

    distance: f64,
    id: ID,

    pub fn order(self: Self) f64 {
        return self.distance;
    }
};
