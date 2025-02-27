#ifndef PARTICLE
#define PARTICLE

class Particle {
	public:
		double mass;
		double radius;
		sf::Vector2f pos;
        sf::Vector2f vel;
        int hue;

		Particle(const double mass, const double radius, const sf::Vector2f pos, const sf::Vector2f vel, const int hue) {
			this->mass = mass;
            this->radius = radius;
            this->pos = pos;
            this->vel = vel;
            this->hue = hue;
		}

		void update() {
            // 		self.pos += self.vel
// 		if self.pos.x < self.radius:
// 			self.vel.x *= -1
// 			self.pos.x = self.radius
// 		elif self.pos.x > SCENE_WIDTH - self.radius:
// 			self.vel.x *= -1
// 			self.pos.x = SCENE_WIDTH - self.radius
// 		if self.pos.y < self.radius:
// 			self.vel.y *= -1
// 			self.pos.y = self.radius
// 		elif self.pos.y > SCENE_HEIGHT - self.radius:
// 			self.vel.y *= -1
// 			self.pos.y = SCENE_HEIGHT - self.radius
		}

		void collide(Particle other) {
// 		d = self.pos.distance_to(other.pos) + 2

// 		if d > self.radius + other.radius:
// 			return

// 		impact_vector = other.pos - self.pos

// 		# Push apart particles so they aren't overlapping
// 		overlap = d - (self.radius + other.radius)
// 		delta_pos = impact_vector.copy()
// 		delta_pos.scale_to_length(overlap * 0.5)
// 		self.pos += delta_pos
// 		other.pos -= delta_pos

// 		# Correct the distance
// 		d = self.radius + other.radius
// 		impact_vector.scale_to_length(d)

// 		# Numerators for updating this particle (a) and other particle (b), and denominator for both
// 		num_a = (other.vel - self.vel).dot(impact_vector) * 2 * other.mass
// 		num_b = (other.vel - self.vel).dot(impact_vector) * -2 * self.mass
// 		den = (self.mass + other.mass) * d * d

// 		# Update this particle (a)
// 		delta_v_a = impact_vector.copy()
// 		delta_v_a *= num_a / den
// 		self.vel += delta_v_a

// 		# Update other particle (b)
// 		delta_v_b = impact_vector.copy()
// 		delta_v_b *= num_b / den
// 		other.vel += delta_v_b
		}
};

#endif
