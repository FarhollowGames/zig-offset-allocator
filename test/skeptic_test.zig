//! The skeptic's tests: inputs and checks the other tests do not make.
//!
//! - Differential runs at edges the configured runs do not reach on
//!   purpose (exact fits with one node left, zero-size storms, 300,000
//!   32-bit nodes run out, all 65,536 16-bit nodes over the full u32
//!   space), against the original, every field compared.
//! - `Ranges.storageReport().largest_free_region` against the core's, in
//!   bytes: no other test compares it outside two fixed states.
//! - `init` out of memory for `Allocator16` and `Ranges16`.
//! - A `Ranges` generation after 2^32 allocate/free cycles of one node, set
//!   directly: it wraps without a panic, and the documented limit holds.

const std = @import("std");
const testing = std.testing;
const oa = @import("offset_allocator");

const Cxx = opaque {};

fn Reference(comptime prefix: []const u8) type {
    return struct {
        const Handle = *Cxx;
        fn ext(comptime name: []const u8, comptime T: type) *const T {
            return @extern(*const T, .{ .name = prefix ++ name });
        }
        const create = ext("create", fn (u32, u32) callconv(.c) ?Handle);
        const destroy = ext("destroy", fn (Handle) callconv(.c) void);
        const reset = ext("reset", fn (Handle) callconv(.c) void);
        const allocate = ext("allocate", fn (Handle, u32, *u32, *u32) callconv(.c) void);
        const free = ext("free", fn (Handle, u32, u32) callconv(.c) void);
        const allocation_size = ext("allocation_size", fn (Handle, u32, u32) callconv(.c) u32);
        const storage_report = ext("storage_report", fn (Handle, *[2]u32) callconv(.c) void);
        const storage_report_full = ext(
            "storage_report_full",
            fn (Handle, *[2 * oa.num_leaf_bins]u32) callconv(.c) void,
        );
        const state = ext(
            "state",
            fn (Handle, *[5]u32, *[oa.num_top_bins]u8, *[oa.num_leaf_bins]u32) callconv(.c) void,
        );
        const node = ext("node", fn (Handle, u32, *[7]u32) callconv(.c) void);
        const free_node = ext("free_node", fn (Handle, u32) callconv(.c) u32);
    };
}

const Ref32 = Reference("oa32_");
const Ref16 = Reference("oa16_");

fn Diff(comptime Allocator: type, comptime Ref: type) type {
    return struct {
        zig: Allocator,
        ref: Ref.Handle,
        live: std.ArrayList(Allocator.Allocation) = .empty,

        const Self = @This();

        fn init(size: u32, max_allocs: u32) !Self {
            return .{
                .zig = try Allocator.init(testing.allocator, size, max_allocs),
                .ref = Ref.create(size, max_allocs) orelse return error.OutOfMemory,
            };
        }

        fn deinit(self: *Self) void {
            self.zig.deinit(testing.allocator);
            Ref.destroy(self.ref);
            self.live.deinit(testing.allocator);
        }

        fn allocate(self: *Self, size: u32) !bool {
            var offset: u32 = undefined;
            var metadata: u32 = undefined;
            Ref.allocate(self.ref, size, &offset, &metadata);
            if (self.zig.allocate(size)) |a| {
                try testing.expectEqual(offset, a.offset);
                try testing.expectEqual(metadata, a.metadata);
                try self.live.append(testing.allocator, a);
                return true;
            } else |_| {
                try testing.expectEqual(oa.no_space, offset);
                return false;
            }
        }

        fn free(self: *Self, index: usize) void {
            const a = self.live.swapRemove(index);
            Ref.free(self.ref, a.offset, a.metadata);
            self.zig.free(a);
        }

        fn reset(self: *Self) void {
            Ref.reset(self.ref);
            self.zig.reset();
            self.live.clearRetainingCapacity();
        }

        fn expectSame(self: *Self, nodes: bool) !void {
            var report: [2]u32 = undefined;
            Ref.storage_report(self.ref, &report);
            const zr = self.zig.storageReport();
            try testing.expectEqual(report[0], zr.total_free_space);
            try testing.expectEqual(report[1], zr.largest_free_region);

            var scalars: [5]u32 = undefined;
            var used_bins: [oa.num_top_bins]u8 = undefined;
            var bin_indices: [oa.num_leaf_bins]u32 = undefined;
            Ref.state(self.ref, &scalars, &used_bins, &bin_indices);
            const z = &self.zig;
            try testing.expectEqual(scalars, [5]u32{ z.size, z.max_allocs, z.free_storage, z.used_bins_top, z.free_offset });
            try testing.expectEqual(used_bins, z.used_bins);
            for (bin_indices, z.bin_indices) |r, zz| try testing.expectEqual(r, zz);
            if (!nodes) return;
            // O(free regions), so only with the nodes.
            var full: [2 * oa.num_leaf_bins]u32 = undefined;
            Ref.storage_report_full(self.ref, &full);
            for (self.zig.storageReportFull().free_regions, 0..) |region, i| {
                try testing.expectEqual(full[2 * i], region.size);
                try testing.expectEqual(full[2 * i + 1], region.count);
            }
            for (self.live.items) |a| {
                try testing.expectEqual(Ref.allocation_size(self.ref, a.offset, a.metadata), z.allocationSize(a));
            }
            for (z.nodes, z.free_nodes, 0..) |n, f, i| {
                var r: [7]u32 = undefined;
                Ref.node(self.ref, @intCast(i), &r);
                const mine = [7]u32{
                    n.data_offset,   n.data_size,     n.bin_list_prev,      n.bin_list_next,
                    n.neighbor_prev, n.neighbor_next, @intFromBool(n.used),
                };
                try testing.expectEqual(r, mine);
                try testing.expectEqual(Ref.free_node(self.ref, @intCast(i)), f);
            }
        }
    };
}

test "skeptic: exact fits and frees with one node left and none left, both widths" {
    inline for (.{ .{ oa.Allocator, Ref32 }, .{ oa.Allocator16, Ref16 } }) |pair| {
        for ([_]u32{ 3, 4, 5 }) |nodes| {
            var d = try Diff(pair[0], pair[1]).init(30, nodes);
            defer d.deinit();
            // Splits until one node is left, then exact fits (no node
            // needed), then out of nodes, then frees in every position.
            for ([_]u32{ 10, 10, 10, 0, 5, 0 }) |size| {
                _ = try d.allocate(size);
                try d.expectSame(true);
            }
            while (d.live.items.len > 0) {
                d.free(d.live.items.len / 2);
                try d.expectSame(true);
                _ = try d.allocate(0);
                try d.expectSame(true);
                if (d.live.items.len > 0) {
                    d.free(0);
                    try d.expectSame(true);
                }
            }
        }
        // The whole u32 space: bin 239 exactly, its remainder, and requests
        // that round up to bin 240.
        var d = try Diff(pair[0], pair[1]).init(0xffff_ffff, 3);
        defer d.deinit();
        for ([_]u32{ 0xf000_0001, 0xf000_0000, 0x0fff_ffff, 0, 1, 0xffff_ffff }) |size| {
            _ = try d.allocate(size);
            try d.expectSame(true);
        }
        while (d.live.items.len > 0) {
            d.free(0);
            try d.expectSame(true);
        }
    }
}

test "skeptic: zero-size storms in a tiny space, both widths" {
    inline for (.{ .{ oa.Allocator, Ref32 }, .{ oa.Allocator16, Ref16 } }) |pair| {
        for (0..50) |seed| {
            var prng: std.Random.DefaultPrng = .init(0x5ce7_0000 + seed);
            const random = prng.random();
            const total = random.uintAtMost(u32, 12);
            const nodes = random.intRangeAtMost(u32, 2, 40);
            var d = try Diff(pair[0], pair[1]).init(total, nodes);
            defer d.deinit();
            for (0..3000) |_| {
                const roll = random.uintLessThan(u32, 100);
                if (roll == 0) {
                    d.reset();
                } else if (roll < 50 and d.live.items.len > 0) {
                    d.free(random.uintLessThan(usize, d.live.items.len));
                } else {
                    _ = try d.allocate(random.uintAtMost(u32, 2));
                }
                try d.expectSame(true);
            }
        }
    }
}

test "skeptic: 300,000 32-bit nodes run out, then free in random order" {
    var prng: std.Random.DefaultPrng = .init(0x5ce7);
    const random = prng.random();
    var d = try Diff(oa.Allocator, Ref32).init(1 << 22, 300_000);
    defer d.deinit();
    var failures: u32 = 0;
    var calls: u32 = 0;
    while (failures < 64) : (calls += 1) {
        failures = if (try d.allocate(random.uintAtMost(u32, 8))) 0 else failures + 1;
        if (calls % 4096 == 0) d.free(random.uintLessThan(usize, d.live.items.len));
        try d.expectSame(calls % 50_000 == 0);
    }
    try d.expectSame(true);
    while (d.live.items.len > 0) {
        d.free(random.uintLessThan(usize, d.live.items.len));
        try d.expectSame(d.live.items.len % 50_000 == 0);
    }
    try d.expectSame(true);
}

test "skeptic: all 65,536 16-bit nodes over the whole u32 space" {
    var prng: std.Random.DefaultPrng = .init(0x5ce16);
    const random = prng.random();
    var d = try Diff(oa.Allocator16, Ref16).init(0xffff_ffff, 65536);
    defer d.deinit();
    for (0..2) |_| {
        var failures: u32 = 0;
        var calls: u32 = 0;
        while (failures < 64) : (calls += 1) {
            const size = switch (random.uintLessThan(u32, 8)) {
                0 => random.uintAtMost(u32, 0xffff_ffff),
                1 => random.uintAtMost(u32, 1 << 20),
                else => random.uintAtMost(u32, 64),
            };
            failures = if (try d.allocate(size)) 0 else failures + 1;
            if (d.live.items.len > 0 and random.uintLessThan(u32, 4) == 0) {
                d.free(random.uintLessThan(usize, d.live.items.len));
            }
            try d.expectSame(calls % 20_000 == 0);
        }
        try d.expectSame(true);
        while (d.live.items.len > 0) {
            d.free(random.uintLessThan(usize, d.live.items.len));
            try d.expectSame(d.live.items.len % 20_000 == 0);
        }
        try d.expectSame(true);
        d.reset();
        try d.expectSame(true);
    }
}

test "skeptic: Ranges.storageReport largest_free_region is the core's, in bytes" {
    // ranges_test.zig compares only total_free_space against its model; the
    // largest region is checked in two fixed states, where it equals the
    // total. A Ranges that reported the total as the largest region passes
    // every other test.
    inline for (.{ oa.Ranges, oa.Ranges16 }) |Ranges| {
        for (0..200) |seed| {
            var prng: std.Random.DefaultPrng = .init(0x5ce_4a46 + seed);
            const random = prng.random();
            const shift = random.uintAtMost(u5, 20);
            const unit = @as(u32, 1) << shift;
            const units = random.intRangeAtMost(u64, 1, 1 << 20);
            var ranges = try Ranges.init(testing.allocator, units << shift, random.intRangeAtMost(u32, 3, 256), unit);
            defer ranges.deinit(testing.allocator);
            var live: std.ArrayList(Ranges.Handle) = .empty;
            defer live.deinit(testing.allocator);
            for (0..300) |_| {
                if (live.items.len > 0 and random.boolean()) {
                    ranges.free(live.swapRemove(random.uintLessThan(usize, live.items.len)));
                } else if (ranges.allocate(random.uintAtMost(u64, (units << shift) / 8))) |h| {
                    try live.append(testing.allocator, h);
                } else |_| {}
                const core = ranges.core.storageReport();
                const bytes = ranges.storageReport();
                try testing.expectEqual(@as(u64, core.largest_free_region) << shift, bytes.largest_free_region);
                try testing.expectEqual(@as(u64, core.total_free_space) << shift, bytes.total_free_space);
                try testing.expect(bytes.largest_free_region <= bytes.total_free_space);
                // The report's promise: a request of the largest region's
                // size fits while a node is left.
                if (bytes.largest_free_region > 0) {
                    const h = try ranges.allocate(bytes.largest_free_region);
                    ranges.free(h);
                }
            }
        }
    }
}

test "skeptic: init out of memory frees what it took, Allocator16 and Ranges16" {
    for (0..2) |fail_index| {
        var failing: std.testing.FailingAllocator = .init(testing.allocator, .{ .fail_index = fail_index });
        try testing.expectError(error.OutOfMemory, oa.Allocator16.init(failing.allocator(), 100, 16));
    }
    for (0..3) |fail_index| {
        var failing: std.testing.FailingAllocator = .init(testing.allocator, .{ .fail_index = fail_index });
        try testing.expectError(error.OutOfMemory, oa.Ranges16.init(failing.allocator(), 1 << 20, 16, 256));
    }
}

test "skeptic: a Ranges generation wraps at 2^32 without a panic, as documented" {
    var ranges = try oa.Ranges.init(testing.allocator, 1 << 20, 16, 256);
    defer ranges.deinit(testing.allocator);
    // Allocating and freeing one size gives the same node back every time,
    // so its generation moves on by one per cycle.
    const oldest = try ranges.allocate(256);
    ranges.free(oldest);
    var current = try ranges.allocate(256);
    try testing.expectEqual(oldest.node, current.node);
    // Where 2^32 - 2 more cycles would leave the generation.
    ranges.generations[current.node] = std.math.maxInt(u32);
    current.generation = std.math.maxInt(u32);
    const newest_stale = current;
    // The free wraps the generation to 0 without an overflow panic.
    ranges.free(current);
    current = try ranges.allocate(256);
    try testing.expectEqual(oldest.node, current.node);
    try testing.expectEqual(@as(u32, 0), current.generation);
    // The handle freed one cycle ago is stale.
    try testing.expect(!ranges.isLive(newest_stale));
    // The documented limit: the handle freed 2^32 cycles ago has the
    // current generation again, so it reads as live, for this one cycle.
    try testing.expect(ranges.isLive(oldest));
    ranges.free(current);
    try testing.expect(!ranges.isLive(oldest));
}
