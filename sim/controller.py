"""Orbi's panel-push gait as a feedback controller on the MuJoCo model.

Same idea as the page's simulate(): a panel on the low side extends, meets the floor and pushes;
the sphere rolls away and coasts. Here the controller only sets the 14 extension targets; whether
the body moves is up to the physics. Each push is aimed at the velocity error (wanted minus
actual), so the same rule accelerates, steers and brakes. At the target the three lowest panels
come down as a tripod.
"""
import numpy as np

from build_model import AXES, R, STROKE, contact_ext

AX = np.array(AXES)


class Terrain:
    """Top of the static terrain under a point, read from the compiled model (boxes are axis-aligned,
    capsules lie on the floor). The JS port (orbi-physics.js) reads the same arrays the same way."""

    def __init__(self, m):
        self.boxes, self.caps = [], []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != 0:
                continue
            p, s = m.geom_pos[g], m.geom_size[g]
            if m.geom_type[g] == 6:      # box
                self.boxes.append((p[0] - s[0], p[0] + s[0], p[1] - s[1], p[1] + s[1], p[2] + s[2]))
            elif m.geom_type[g] == 3:    # capsule: axis = local z
                w, x, y, z = m.geom_quat[g]
                ax, ay = 2 * (x * z + w * y), 2 * (y * z - w * x)
                self.caps.append((p[0] - ax * s[1], p[1] - ay * s[1], p[0] + ax * s[1], p[1] + ay * s[1], s[0]))

    def height(self, x, y):
        h = 0.0
        for x0, x1, y0, y1, top in self.boxes:
            if x0 <= x <= x1 and y0 <= y <= y1 and top > h:
                h = top
        for ax, ay, bx, by, c in self.caps:
            vx, vy = bx - ax, by - ay
            t = min(1.0, max(0.0, ((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy)))
            dd = (x - ax - vx * t) ** 2 + (y - ay - vy * t) ** 2
            if dd < c * c:
                h = max(h, c + (c * c - dd) ** 0.5)
        return h


class Gait:
    def __init__(self, model, data, pad_speed=1.0, cruise=0.9, approach=0.9, deadband=0.22,
                 push_time=0.30, cool=0.06, score_min=0.5, arrive=0.12,
                 lead=0.06, depth_gain=0.12, depth_min=0.02, depth_max=0.14):
        self.m, self.d = model, data
        self.bid = model.body("shell").id
        self.pad_speed = pad_speed        # ASSUMED max panel extension speed, m/s
        self.cruise, self.approach, self.deadband = cruise, approach, deadband
        self.push_time, self.cool_time, self.score_min, self.arrive = push_time, cool, score_min, arrive
        self.lead, self.depth_gain, self.depth_min, self.depth_max = lead, depth_gain, depth_min, depth_max
        self.depth = 0.0
        self.cmd = np.zeros(14)
        self.push, self.push_t, self.cool = -1, 0.0, 0.0
        self.push_dir = np.zeros(2)
        self.target = None                # (x, y) world
        self.stop_at = True
        self.holding = False
        self.pushes = 0
        self.stand_lift = 0.03
        self.stance = np.zeros(14)
        self.latched = False
        self.climb, self.climb_speed, self.zone_kh, self.zone_kd, self.zone_min, self.lever_len = True, 0.35, 0.8, 0.6, 0.5, 1.2
        self.origin, self.carrot, self.climb_steer = None, 0.5, 1.5
        self.jam, self.jam_max, self.jam_gain, self.jam_decay = 0.0, 1.4, 0.5, 2.0
        self.prop, self.prop_time, self.rest, self.rest_time = 0.0, 2.5, 0.0, 0.7
        self.hover, self.latch_tol, self.latch_speed = -0.003, 0.006, 0.1   # hover < 0: feet drag lightly and brake
        self.terrain, self.look, self.rise_min, self.rise_max = None, (0.5, 0.8, 1.1, 1.4), 0.03, 0.2
        self.stall, self.stall_time, self.lever_t, self.push_lever, self.levers = 0.0, 0.35, 0.0, False, 0
        # a push is a measured FORCE, not a depth: the panel is commanded just far enough past contact
        # that the servo presses with a fraction `lift` of the rover's weight (vertical part), so the
        # sphere stays on the floor and rolls instead of being thrown up
        self.weight = float(model.body_subtreemass[self.bid]) * 9.81
        self.kp = float(model.actuator_gainprm[0, 0])
        self.lift_gain, self.lift_min, self.lift_max, self.lift_near = 1.0, 0.25, 0.7, 0.1
        self.kh, self.kd, self.vz_cut = 0.62, 0.78, 0.5
        self.force_push = True
        self.zref, self.drop_tol, self.lowering, self.catch = np.inf, 0.02, False, True
        self.v_end, self.brake = None, 1.2   # speed to carry through the target when not stopping there (None: cruise)
        self.lower_speed = 0.25           # panels holding the body up retract no faster than this: lower it, never drop it

    def step(self, ground_z=0.0):
        m, d, dt = self.m, self.d, self.m.opt.timestep
        p, v = d.qpos[:3], d.qvel[:3]          # free joint: world position and linear velocity
        if self.terrain is not None:
            ground_z = self.terrain.height(p[0], p[1])
        Rm = d.xmat[self.bid].reshape(3, 3)
        aw = AX @ Rm.T
        # where each panel will point after the actuator's lead time (free-joint angular velocity is body-local)
        ap = aw + self.lead * np.cross(Rm @ d.qvel[3:6], aw)
        grounded = p[2] - ground_z < R + 0.02
        h = p[2] - ground_z                    # centre height over the local ground
        down = -aw[:, 2]
        ext_c = contact_ext(down, h)           # extension at which each panel meets flat ground
        want = np.zeros(14)
        rate = self.pad_speed
        if self.target is not None:
            dvec = np.array([self.target[0] - p[0], self.target[1] - p[1]])
            dist = float(np.linalg.norm(dvec))
            if self.origin is not None and dist > self.carrot:
                # follow the line from where this leg began: aim at a point `carrot` ahead of the
                # nearest point on it, so sideways drift is steered out instead of carried along
                o = np.array(self.origin, float)
                seg = np.array(self.target, float) - o
                L = float(np.linalg.norm(seg))
                if L > 1e-6:
                    seg /= L
                    s = float(np.clip((np.array(p[:2]) - o) @ seg + self.carrot, 0.0, L))
                    dvec = o + seg * s - np.array(p[:2])
            dn = float(np.linalg.norm(dvec))
            dhat = dvec / dn if dn > 1e-6 else np.zeros(2)
            vh = np.array(v[:2])
            sp = float(np.linalg.norm(vh))
            vd = min(self.cruise, self.approach * dist) if self.stop_at else self.cruise
            if not self.stop_at and self.v_end is not None:
                vd = min(vd, (self.v_end ** 2 + 2 * self.brake * dist) ** 0.5)   # ease into a corner
            v_des = dhat * vd
            err = v_des - vh
            en = float(np.linalg.norm(err))
            near = self.stop_at and dist < 0.6
            if self.stop_at and (dist < 0.3 if self.holding else dist < self.arrive and sp < 0.3):
                # stand, in two beats. 1) hover: every panel that can reach comes down to just above the
                # floor, following the orientation while the sphere is still creeping. 2) plant: once they
                # are all there, latch lengths that lift the centre by `stand_lift`. Fixed legs make a rigid
                # stance; legs that kept following the orientation would just be a bigger ball.
                if not self.holding:
                    self.latched = False
                self.holding, self.push = True, -1
                if not self.latched:
                    feet = (down > 0.35) & (ext_c < 0.9 * STROKE)
                    want = np.where(feet, np.maximum(contact_ext(down, R) - self.hover, 0.0), 0.0)
                    if feet.sum() >= 3 and np.abs(d.qpos[7:21] - want)[feet].max() < self.latch_tol and sp < self.latch_speed:
                        self.stance = np.where(feet, contact_ext(down, R + self.stand_lift), 0.0)
                        self.latched = True
                if self.latched:
                    want = self.stance.copy()
            else:
                self.holding = False
                self.cool -= dt
                # blocked? (wanting to go, not going) -> lever mode: full-stroke pushes from low and behind
                along = float(vh @ dhat)
                # stalled = hardly moving at all (not merely moving sideways: that case is steering, and
                # full-stroke pushes toward the target would then feed an orbit)
                self.stall = self.stall + dt if (sp < 0.15 and along < 0.15 and dist > 0.4) else max(0.0, self.stall - 2 * dt)
                if self.stall > self.stall_time:
                    if self.lever_t <= 0:
                        self.levers += 1
                    self.lever_t = self.lever_len
                self.lever_t -= dt
                lever = self.lever_t > 0
                # a climbable rise just ahead (what the range sensors would see): climb it on purpose
                if self.terrain is not None and dist > 1e-6:
                    rise = max(self.terrain.height(p[0] + dhat[0] * k * R, p[1] + dhat[1] * k * R) for k in self.look) - ground_z
                    if self.rise_min < rise < self.rise_max:
                        lever = True
                # propped up and going nowhere for a while: let go, drop, and start the climb again
                self.rest -= dt
                if lever and self.climb and self.rest <= 0:
                    self.prop = self.prop + dt if sp < 0.05 else 0.0
                    if self.prop > self.prop_time:
                        self.prop, self.jam, self.rest = 0.0, 0.0, self.rest_time
                if self.rest > 0:
                    lever = False
                if lever and self.climb:
                    pass
                elif self.push < 0 and self.cool <= 0 and en > (0.10 if near else self.deadband) and (grounded or lever):
                    eh = err / en
                    # the panel that points most nearly down and against the push we want
                    if lever:
                        eh = dhat
                    kh, kd = (0.80, 0.60) if lever else (self.kh, self.kd)
                    score = -(ap[:, 0] * eh[0] + ap[:, 1] * eh[1]) * kh - ap[:, 2] * kd
                    score[ext_c > 0.9 * STROKE] = -1
                    best = int(np.argmax(score))
                    if score[best] > self.score_min:
                        self.push, self.push_t, self.push_dir = best, self.push_time, eh
                        self.depth = float(np.clip(self.depth_gain * en, 0.006 if near else self.depth_min, self.depth_max))
                        self.push_lever = lever
                        if self.force_push:
                            self.depth = float(np.clip(self.lift_gain * en, self.lift_near if near else self.lift_min, self.lift_max)) * self.weight / self.kp
                        if lever:
                            self.depth, self.push_t = STROKE, 0.6
                        self.pushes += 1
                if self.push >= 0:
                    dep = self.depth / max(down[self.push], 0.3) if self.force_push and not self.push_lever else self.depth
                    want[self.push] = min(STROKE, ext_c[self.push] + dep)
                    self.push_t -= dt
                    closed = float(err @ self.push_dir) < 0.03          # this push has done its job
                    turned = aw[self.push, :2] @ self.push_dir > -0.08   # rolled past: it would now push the wrong way
                    soft_end = not self.push_lever and (closed or turned or v[2] > self.vz_cut)
                    if self.push_t <= 0 or soft_end or d.qpos[7 + self.push] > 0.97 * STROKE:
                        self.push, self.cool = -1, self.cool_time
                # climb: while blocked, every panel pointing down-and-back extends, slowly, as far as it
                # goes. The sphere pivots up over the edge it is leaning on; as it turns, panels leave
                # the zone and retract and the next ones enter it (a wheel of extending spokes).
                if lever and self.climb:
                    lat = vh - dhat * along
                    st = dhat - self.climb_steer * lat
                    st /= max(float(np.linalg.norm(st)), 1e-6)
                    # jammed (propped and still): let panels further forward join in until it moves again
                    self.jam = min(self.jam + dt, self.jam_max) if sp < 0.05 else max(0.0, self.jam - self.jam_decay * dt)
                    zmin = self.zone_min - self.jam_gain * max(0.0, self.jam - 0.4)
                    zone = (-(aw[:, 0] * st[0] + aw[:, 1] * st[1]) * self.zone_kh - aw[:, 2] * self.zone_kd) > zmin
                    want = np.where(zone, STROKE, 0.0)
                    rate = self.climb_speed
                    self.push = -1
                # over a drop (rolled past an edge, or let go of a climb): feet come down to the floor
                # below and lower the body at `lower_speed`, instead of letting it fall
                self.lowering = False
                if self.catch and not (lever and self.climb) and h > R + self.drop_tol:
                    self.zref = max(R, min(self.zref, h) - self.lower_speed * dt)
                    zc = ground_z + self.zref
                    for i in np.nonzero(down > 0.35)[0]:
                        gz = ground_z
                        if self.terrain is not None and np.isfinite(ext_c[i]):
                            reach = R + min(ext_c[i], STROKE)
                            gz = self.terrain.height(p[0] + aw[i, 0] * reach, p[1] + aw[i, 1] * reach)
                        e = float(contact_ext(down[i], zc - gz))
                        if e < 0.97 * STROKE:
                            want[i] = e
                    self.push, self.lowering = -1, True
                else:
                    self.zref = np.inf
        # panels move no faster than pad_speed
        down_rate = self.pad_speed if grounded else self.lower_speed
        self.cmd += np.clip(want - self.cmd, -down_rate * dt, rate * dt)
        d.ctrl[:] = self.cmd
