const Metadata = @import("metadata.zig").Metadata;

pub const Filter = struct {
    attribute: []const u8,
    condition: fn (Metadata) bool,
};
