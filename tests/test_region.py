import aspsim.room.region as region
import numpy as np

def _random_cylinder(rng=None):
    if rng is None:
        rng = np.random.default_rng()
    radius = rng.uniform(0.5, 2)
    height = rng.uniform(0.5, 2)
    center = rng.uniform(-1, 1, size=3)
    point_spacing = rng.uniform(0.1, 0.5, size=3)
    cylinder = region.Cylinder(radius, height, center, point_spacing, rng=rng)
    return cylinder

def _random_cuboid(rng=None):
    if rng is None:
        rng = np.random.default_rng()
    side_lengths = rng.uniform(0.5, 2, size=3)
    center = rng.uniform(-1, 1, size=3)
    point_spacing = rng.uniform(0.1, 0.5, size=3)
    cuboid = region.Cuboid(side_lengths, center, point_spacing, rng=rng)
    return cuboid

def _random_ball(rng=None):
    if rng is None:
        rng = np.random.default_rng()
    radius = rng.uniform(0.5, 2)
    center = rng.uniform(-1, 1, size=3)
    point_spacing = rng.uniform(0.1, 0.5, size=3)
    ball = region.Ball(radius, center, point_spacing, rng=rng)
    return ball


def _random_rectangle(rng=None):
    if rng is None:
        rng = np.random.default_rng()
    side_lengths = rng.uniform(0.5, 2, size=2)
    center = rng.uniform(-1, 1, size=3)
    point_spacing = rng.uniform(0.1, 0.5, size=2)
    rect = region.Rectangle(side_lengths, center, point_spacing, rng=rng)
    return rect

def test_cylinder_equally_spaced_points_returns_points_within_region():
    cyl = _random_cylinder()
    points = cyl.equally_spaced_points()

    for i in range(points.shape[0]):
        point = points[i,:]
        assert (point[0]-cyl.center[0]) ** 2 + (point[1]-cyl.center[1]) ** 2 <= cyl.radius ** 2
        assert cyl.center[2] - cyl.height / 2 <= point[2] <= cyl.center[2] + cyl.height / 2
        assert cyl.is_in_region(point[None,:])

def test_cylinder_sample_points_returns_points_within_region():
    cyl = _random_cylinder()
    num_to_sample = 1000
    points = cyl.sample_points(num_to_sample)

    for i in range(points.shape[0]):
        point = points[i,:]
        assert (point[0]-cyl.center[0]) ** 2 + (point[1]-cyl.center[1]) ** 2 <= cyl.radius ** 2
        assert cyl.center[2] - cyl.height / 2 <= point[2] <= cyl.center[2] + cyl.height / 2
        assert cyl.is_in_region(point[None,:])


def test_cylinder_a_random_point_is_always_close_to_equally_spaced_point():
    cyl = _random_cylinder()
    points = cyl.equally_spaced_points()

    num_to_sample = 1000
    points_random = cyl.sample_points(num_to_sample)

    for i in range(points_random.shape[0]):
        diff = points - points_random[i:i+1,:]
        min_distance = np.min(np.abs(diff), axis=0)
        assert np.all(min_distance < cyl.point_spacing)



def test_cuboid_equally_spaced_points_returns_points_within_region():
    cb = _random_cuboid()
    points = cb.equally_spaced_points()

    for i in range(points.shape[0]):
        point = points[i,:]
        assert np.all(cb.center - cb.side_lengths / 2 <= point)
        assert np.all(point <= cb.center + cb.side_lengths / 2)
        assert cb.is_in_region(point[None,:])

def test_cuboid_sample_points_returns_points_within_region():
    cb = _random_cuboid()
    num_to_sample = 1000
    points = cb.sample_points(num_to_sample)

    for i in range(points.shape[0]):
        point = points[i,:]
        assert np.all(cb.center - cb.side_lengths / 2 <= point)
        assert np.all(point <= cb.center + cb.side_lengths / 2)
        assert cb.is_in_region(point[None,:])

def test_cuboid_a_random_point_is_always_close_to_equally_spaced_point():
    cb = _random_cuboid()
    points = cb.equally_spaced_points()

    num_to_sample = 1000
    points_random = cb.sample_points(num_to_sample)

    for i in range(points_random.shape[0]):
        diff = points - points_random[i:i+1,:]
        min_distance = np.min(np.abs(diff), axis=0)
        assert np.all(min_distance < cb.point_spacing)




def test_ball_equally_spaced_points_returns_points_within_region():
    ball = _random_ball()
    points = ball.equally_spaced_points()

    for i in range(points.shape[0]):
        point = points[i,:]
        assert np.all(np.linalg.norm(point - ball.center, axis=-1) <= ball.radius) 
        assert ball.is_in_region(point[None,:])

def test_ball_sample_points_returns_points_within_region():
    ball = _random_ball()
    num_to_sample = 1000
    points = ball.sample_points(num_to_sample)

    for i in range(points.shape[0]):
        point = points[i,:]
        assert np.all(np.linalg.norm(point - ball.center, axis=-1) <= ball.radius) 
        assert ball.is_in_region(point[None,:])

def test_ball_a_random_point_is_always_close_to_equally_spaced_point():
    ball = _random_ball()
    points = ball.equally_spaced_points()

    num_to_sample = 1000
    points_random = ball.sample_points(num_to_sample)

    for i in range(points_random.shape[0]):
        diff = points - points_random[i:i+1,:]
        min_distance = np.min(np.abs(diff), axis=0)
        assert np.all(min_distance < ball.point_spacing)




def test_rectangle_equally_spaced_points_returns_points_within_region():
    rect = _random_rectangle()
    points = rect.equally_spaced_points()

    for i in range(points.shape[0]):
        point = points[i,:]
        side_len_3d = np.concatenate((rect.side_lengths, [0]))
        assert np.all(rect.center - side_len_3d / 2 <= point)
        assert np.all(point <= rect.center + side_len_3d / 2)
        assert rect.is_in_region(point[None,:])

def test_rectangle_sample_points_returns_points_within_region():
    rect = _random_rectangle()
    num_to_sample = 1000
    points = rect.sample_points(num_to_sample)

    for i in range(points.shape[0]):
        point = points[i,:]
        side_len_3d = np.concatenate((rect.side_lengths, [0]))
        assert np.all(rect.center - side_len_3d / 2 <= point)
        assert np.all(point <= rect.center + side_len_3d / 2)
        assert rect.is_in_region(point[None,:])

def test_rectangle_a_random_point_is_always_close_to_equally_spaced_point():
    rect = _random_rectangle()
    points = rect.equally_spaced_points()

    num_to_sample = 1000
    points_random = rect.sample_points(num_to_sample)

    for i in range(points_random.shape[0]):
        diff = points - points_random[i:i+1,:]
        min_distance = np.min(np.abs(diff), axis=0)
        assert np.all(min_distance[:2] < rect.point_spacing)