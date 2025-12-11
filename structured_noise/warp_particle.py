from utils import *

@ti.func
def is_in_bound(u, v, n, m):
    result = False
    if u >= 0 and v >= 0 and u < n and v < m:
        result = True
    return result

@ti.kernel
def particle_warp_kernel(map_field: ti.template(), noise_field: ti.template(), buffer_field: ti.template(),
                ticket_serial_field: ti.template(), master_field: ti.template(), area_field: ti.template(),
                pixel_area_field: ti.template()):
   
    img_n = noise_field.shape[0]
    img_m = noise_field.shape[1]
    for i, j in noise_field:
        pixel_area_field[i,j] *= 0 # # clear pixel area
        buffer_field[i,j] *= 0 # clear buffer
        ticket_serial_field[i,j] *= 0 # clear ticket serial 
        
    for i, j in noise_field:
        raveled_index = ravel_index(i, j, img_n, img_m)
        warped_pos = map_field[i, j]-0.5 # notice the -0.5, we are aligning with the grid
        lower_corner = ti.math.floor(warped_pos)
        frac = warped_pos - lower_corner

        lower_x, lower_y = int(lower_corner.x), int(lower_corner.y)
        upper_x, upper_y = lower_x + 1, lower_y + 1
        # handle lower_x, lower_y
        if is_in_bound(lower_x, lower_y, img_n, img_m):
            frac_lower_lower = (1.-frac.x) * (1.-frac.y) # bilinear weight for lower-lower
            if frac_lower_lower > 0.:
                ticket_lower_lower = ti.atomic_add(ticket_serial_field[lower_x, lower_y], 1)
                master_field[lower_x, lower_y, ticket_lower_lower] = raveled_index
                area_field[lower_x, lower_y, ticket_lower_lower] = frac_lower_lower  
        # handle upper_x, upper_y
        if is_in_bound(upper_x, upper_y, img_n, img_m):
            frac_upper_upper = frac.x * frac.y # bilinear weight for upper-upper
            if frac_upper_upper > 0.:
                ticket_upper_upper = ti.atomic_add(ticket_serial_field[upper_x, upper_y], 1)
                master_field[upper_x, upper_y, ticket_upper_upper] = raveled_index
                area_field[upper_x, upper_y, ticket_upper_upper] = frac_upper_upper
        # handle lower_x, upper_y
        if is_in_bound(lower_x, upper_y, img_n, img_m):
            frac_lower_upper = (1.-frac.x) * frac.y # bilinear weight for lower-upper
            if frac_lower_upper > 0.:
                ticket_lower_upper = ti.atomic_add(ticket_serial_field[lower_x, upper_y], 1)
                master_field[lower_x, upper_y, ticket_lower_upper] = raveled_index
                area_field[lower_x, upper_y, ticket_lower_upper] = frac_lower_upper   
        # handle upper_x, lower_y
        if is_in_bound(upper_x, lower_y, img_n, img_m):
            frac_upper_lower = frac.x * (1.-frac.y) # bilinear weight for upper-lower
            if frac_upper_lower > 0.:
                ticket_upper_lower = ti.atomic_add(ticket_serial_field[upper_x, lower_y], 1)
                master_field[upper_x, lower_y, ticket_upper_lower] = raveled_index
                area_field[upper_x, lower_y, ticket_upper_lower] = frac_upper_lower

    for u, v in noise_field:
        # raveled_idx = ravel_index(u, v, img_n, img_m)
        # first pass: determine normalization factor (total request) to make sure sum-of-one
        total_request = 0.
        k_idx = 0
        access_record = area_field[u, v, k_idx]
        while access_record > 0.:
            total_request += access_record
            k_idx += 1
            access_record = area_field[u, v, k_idx] # get next
        # second pass: sample and send to source pixels
        if total_request > 0.: # if no request, then don't bother -- no one to send to
            k_idx = 0
            access_record = area_field[u, v, k_idx]
            access_source = master_field[u, v, k_idx]
            past_range = 0.0
            past_value = ti.Vector([0. for _ in ti.static(range(noise_field.n))])
            while access_record > 0.:
                # 
                curr_normalized_request = access_record / total_request
                source_i, source_j = unravel_index(access_source, img_n, img_m) # this is where the request is from
                next_range = past_range + curr_normalized_request
                next_value = sample_brownian_bridge(noise_field[u,v], past_range, next_range, past_value)
                curr_value = next_value - past_value
                past_range = next_range
                past_value = next_value
                buffer_field[source_i, source_j] += curr_value
                pixel_area_field[source_i, source_j] += curr_normalized_request
                area_field[u, v, k_idx] *= 0 # clear area as we go -- an area of 0. means we've reached the end
                k_idx += 1
                access_record = area_field[u, v, k_idx] # get next
                access_source = master_field[u, v, k_idx]
    
    # copy over
    for i, j in noise_field:
        pixel_area = pixel_area_field[i,j]
        if pixel_area > 0.:
            noise_field[i,j] = 1./ti.math.sqrt(pixel_area) * buffer_field[i,j]
        else: # if pixel has not been assigned area
            noise_field[i,j] = get_randn_like(noise_field)


@ti.data_oriented
class ParticleWarper:
    def __init__(self, im_height, im_width, num_noise_channel):
        self.num_noise_channel = num_noise_channel
        #
        self.master_field = ti.field(ti.i32)
        self.area_field = ti.field(ti.f64)
        max_entries = 10000 # max number that a cell is allowed to split into
        dense_size = 8
        self.block = ti.root.pointer(ti.ijk, (math.ceil(im_height/dense_size), math.ceil(im_width/dense_size), math.ceil(max_entries/dense_size)))
        self.pixel = self.block.dense(ti.ijk, (dense_size, dense_size, dense_size))
        self.pixel.place(self.master_field, self.area_field)
        # 
        self.noise_field = ti.Vector.field(self.num_noise_channel, ti.f64, shape=(im_height, im_width))
        fill_noise(self.noise_field)
        self.buffer_field = ti.Vector.field(self.num_noise_channel, ti.f64, shape=(im_height, im_width))
        self.pixel_area_field = ti.field(ti.f64, shape=(im_height, im_width))
        self.ticket_serial_field = ti.field(ti.i32, shape=(im_height, im_width)) # aux variable for order keeping
        self.map_field = ti.Vector.field(2, ti.f64, shape=(im_height, im_width)) #
    
    def set_noise(self, noise_array):
        self.noise_field.from_numpy(noise_array)

    def set_deformation(self, map_array):
        self.map_field.from_numpy(map_array)

    def run(self):
        particle_warp_kernel(self.map_field, self.noise_field, self.buffer_field, self.ticket_serial_field, self.master_field, self.area_field, self.pixel_area_field)