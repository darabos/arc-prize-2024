use crate::tools;
use crate::tools::{Example, Image, Res, Shape, Task, COLORS};
use std::rc::Rc;

fn init_shape_vec(n: usize, vec: &mut Option<Vec<Vec<Rc<Shape>>>>) {
    if vec.is_none() {
        *vec = Some((0..n).map(|_| vec![]).collect());
    }
}

pub fn find_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    init_shape_vec(s.images.len(), &mut s.shapes);
    s.shapes.as_mut().unwrap()[i] = tools::find_shapes_in_image(&s.images[i]);
    Ok(())
}

pub fn find_colorsets(s: &mut SolverState, i: usize) -> Res<()> {
    init_shape_vec(s.images.len(), &mut s.colorsets);
    s.colorsets.as_mut().unwrap()[i] = tools::find_colorsets_in_image(&s.images[i]);
    Ok(())
}

pub fn use_colorsets_as_shapes(s: &mut SolverState) -> Res<()> {
    s.shapes = s.colorsets.take();
    Ok(())
}

pub fn sort_shapes_by_size(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes.as_mut().expect("must have shapes")[i];
    tools::sort_shapes_by_size(shapes);
    Ok(())
}

fn getmut<T>(rc: &mut Rc<T>) -> &mut T {
    Rc::get_mut(rc).expect("rc get_mut failed")
}

fn remap_colors(s: &mut SolverState, i: usize, mapping: &[i32]) {
    s.images[i] = tools::remap_colors_in_image(&s.images[i], mapping);
    if i < s.output_images.len() {
        s.output_images[i] = tools::remap_colors_in_image(&s.output_images[i], mapping);
    }
    if let Some(colorsets) = &mut s.colorsets {
        for shape in colorsets[i].iter_mut() {
            let mut new_shape = (**shape).clone();
            new_shape.recolor(mapping[shape.color() as usize]);
            *shape = Rc::new(new_shape);
        }
    }
    if let Some(shapes) = &mut s.shapes {
        for shape in shapes[i].iter_mut() {
            let mut new_shape = (**shape).clone();
            new_shape.recolor(mapping[shape.color() as usize]);
            *shape = Rc::new(new_shape);
        }
    }
    for saved_shapes in &mut s.saved_shapes {
        for shape in saved_shapes[i].iter_mut() {
            let mut new_shape = (**shape).clone();
            new_shape.recolor(mapping[shape.color() as usize]);
            *shape = Rc::new(new_shape);
        }
    }
    if i == s.images.len() - 1 {
        s.used_colors = tools::get_used_colors(&s.images);
    }
}

/// Renumbers the colors of the image to match the order of the shapes.
/// Modifies the image and the shapes. Returns the mapping.
fn remap_colors_by_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes.as_mut().expect("must have shapes")[i];
    let mut mapping = vec![-1; COLORS.len()];
    for (i, shape) in shapes.iter_mut().enumerate() {
        mapping[shape.color() as usize] = i as i32 + 1;
    }
    remap_colors(s, i, &mapping);
    if s.color_mapping.is_none() {
        s.color_mapping = Some(vec![vec![]; s.images.len()]);
    }
    s.color_mapping.as_mut().unwrap()[i] = mapping;
    Ok(())
}

fn unmap_colors(s: &mut SolverState, i: usize) -> Res<()> {
    let mapping = &s.color_mapping.as_ref().expect("must have color mapping")[i];
    let mut reverse_mapping = vec![-1; COLORS.len()];
    for (i, &c) in mapping.iter().enumerate() {
        if c != -1 {
            reverse_mapping[c as usize] = i as i32;
        }
    }
    remap_colors(s, i, &reverse_mapping);
    if i == s.images.len() - 1 {
        s.color_mapping = None;
    }
    Ok(())
}

/// Returns the first element of each list.
fn get_firsts<T>(vec: &Vec<Vec<T>>) -> Res<Vec<&T>> {
    let mut firsts = vec![];
    for e in vec {
        if e.is_empty() {
            return Err("empty list");
        }
        firsts.push(&e[0]);
    }
    Ok(firsts)
}

fn grow_flowers(s: &mut SolverState) -> Res<()> {
    let shapes = &s.shapes.as_ref().ok_or("must have shapes")?;
    let dots = get_firsts(&shapes)?;
    // let input_pattern = find_pattern_around(&s.images[..s.task.train.len()], &dots);
    let output_pattern = tools::find_pattern_around(&s.output_images, &dots);
    // TODO: Instead of growing each dot, we should filter by the input_pattern.
    s.apply(|s: &mut SolverState, i: usize| {
        let shapes = &s.shapes.as_ref().expect("must have shapes");
        let dots = &shapes[i][0];
        let mut new_image = (*s.images[i]).clone();
        for dot in dots.cells.iter() {
            tools::draw_shape_at(&mut new_image, &dot.pos(), &output_pattern);
        }
        s.images[i] = Rc::new(new_image);
        Ok(())
    })
}

fn save_shapes(s: &mut SolverState) -> Res<()> {
    let shapes = s.shapes.as_ref().ok_or("must have shapes")?;
    s.saved_shapes.push(shapes.clone());
    Ok(())
}

fn load_earlier_shapes(s: &mut SolverState) -> Res<()> {
    if s.saved_shapes.len() < 2 {
        return Err("no saved shapes");
    }
    s.shapes = Some(
        s.saved_shapes
            .get(s.saved_shapes.len() - 2)
            .unwrap()
            .clone(),
    );
    Ok(())
}

fn save_whole_image(s: &mut SolverState) -> Res<()> {
    s.saved_images = s.images.clone();
    Ok(())
}

type Shapes = Vec<Rc<Shape>>;
type ShapesPerExample = Vec<Shapes>;

/// Tracks information while applying operations on all examples at once.
/// Most fields are vectors storing information for each example.
#[derive(Default)]
pub struct SolverState {
    pub task: Rc<Task>,
    pub images: Vec<Rc<Image>>,
    pub saved_images: Vec<Rc<Image>>,
    pub output_images: Vec<Rc<Image>>,
    pub used_colors: Vec<i32>,
    pub shapes: Option<ShapesPerExample>,
    pub saved_shapes: Vec<ShapesPerExample>,
    pub colorsets: Option<ShapesPerExample>,
    pub color_mapping: Option<Vec<Vec<i32>>>,
    pub scale_up: usize,
}

impl SolverState {
    fn new(task: &Task) -> Self {
        let images: Vec<Rc<Image>> = task
            .train
            .iter()
            .chain(task.test.iter())
            .map(|example| Rc::new(example.input.clone()))
            .collect();
        let output_images = task
            .train
            .iter()
            .map(|example| Rc::new(example.output.clone()))
            .collect();
        let used_colors = tools::get_used_colors(&images);
        SolverState {
            task: Rc::new(task.clone()),
            images,
            output_images,
            used_colors,
            ..Default::default()
        }
    }

    fn apply<F>(&mut self, f: F) -> Res<()>
    where
        F: Fn(&mut SolverState, usize) -> Res<()>,
    {
        for i in 0..self.images.len() {
            f(self, i)?;
        }
        Ok(())
    }

    fn get_results(&self) -> Vec<Example> {
        self.images
            .iter()
            .zip(self.task.train.iter().chain(self.task.test.iter()))
            .map(|(image, example)| Example {
                input: example.input.clone(),
                output: (**image).clone(),
            })
            .collect()
    }

    #[allow(dead_code)]
    fn print_shapes(&self) {
        print_shapes(&self.shapes);
    }
    #[allow(dead_code)]
    fn print_saved_shapes(&self) {
        for per_example in &self.saved_shapes {
            for shapes in per_example {
                for shape in shapes {
                    shape.print();
                    println!();
                }
            }
        }
    }
    #[allow(dead_code)]
    fn print_colorsets(&self) {
        print_shapes(&self.colorsets);
    }
    #[allow(dead_code)]
    fn print_images(&self) {
        for image in &self.images {
            tools::print_image(image);
            println!();
        }
    }
}

#[allow(dead_code)]
fn print_shapes(shapes: &Option<Vec<Vec<Rc<Shape>>>>) {
    if let Some(shapes) = shapes {
        for (i, shapes) in shapes.iter().enumerate() {
            println!("Shapes for example {}", i);
            for shape in shapes {
                shape.print();
                println!();
            }
        }
    }
}

fn rotate_used_colors(s: &mut SolverState) -> Res<()> {
    if s.used_colors.is_empty() {
        return Err("no used colors");
    }
    let first_color = s.used_colors[0];
    let n = s.used_colors.len();
    for i in 0..n - 1 {
        s.used_colors[i] = s.used_colors[i + 1];
    }
    s.used_colors[n - 1] = first_color;
    Ok(())
}

fn filter_shapes_by_color(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes.as_mut().expect("must have shapes")[i];
    let color = s.used_colors.get(0).ok_or("no used colors")?;
    s.shapes.as_mut().unwrap()[i] = shapes
        .iter()
        .filter(|shape| shape.color() == *color)
        .cloned()
        .collect();
    Ok(())
}

fn move_shapes_to_saved_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &s.shapes.as_ref().expect("must have shapes")[i];
    let saved_shapes = &s.saved_shapes.last().expect("must have saved shapes")[i];
    let saved_shape = saved_shapes.get(0).ok_or("saved shapes empty")?;
    let mut new_image = (*s.images[i]).clone();
    for shape in shapes {
        new_image = tools::move_shape_to_shape_in_image(&new_image, &shape, &saved_shape)?;
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn resize_image(s: &mut SolverState) -> Res<()> {
    // Find ratio from looking at example outputs.
    let output_size = s.output_images[0].len();
    let input_size = s.images[0].len();
    if output_size % input_size != 0 {
        return Err("output size must be a multiple of input size");
    }
    s.scale_up = output_size / input_size;
    s.apply(|s: &mut SolverState, i: usize| {
        s.images[i] = Rc::new(tools::resize_image(&s.images[i], s.scale_up));
        Ok(())
    })
}

fn save_image_as_shape(s: &mut SolverState, i: usize) -> Res<()> {
    init_shape_vec(s.images.len(), &mut s.shapes);
    s.shapes.as_mut().unwrap()[i] = vec![Rc::new(Shape::from_image(&s.images[i]))];
    Ok(())
}

fn tile_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes.as_mut().expect("must have shapes")[i];
    let current_width = s.images[i][0].len();
    let current_height = s.images[i].len();
    let old_width = current_width / s.scale_up;
    let old_height = current_height / s.scale_up;
    for shape in shapes.iter_mut() {
        let mut new_shape = (**shape).clone();
        new_shape.tile(old_width, s.scale_up, old_height, s.scale_up);
        *shape = Rc::new(new_shape);
    }
    Ok(())
}

fn draw_shape_where_non_empty(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &s.shapes.as_ref().expect("must have shapes")[i];
    let mut new_image = (*s.images[i]).clone();
    for shape in shapes {
        shape.draw_where_non_empty(&mut new_image);
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn delete_shapes_touching_border(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes.as_mut().expect("must have shapes")[i];
    *shapes = shapes
        .iter()
        .filter(|shape| !shape.is_touching_border(&s.images[i]))
        .cloned()
        .collect();
    Ok(())
}

fn recolor_shapes_per_output(s: &mut SolverState) -> Res<()> {
    // Get color from first output_image.
    let first_shape = &s.shapes.as_ref().expect("must have shapes")[0]
        .get(0)
        .ok_or("no shapes")?;
    let color = tools::lookup_in_image(
        &s.output_images[0],
        first_shape.cells[0].x,
        first_shape.cells[0].y,
    )?;
    // Fail the operation if any shape in any output has a different color.
    for (i, image) in s.output_images.iter_mut().enumerate() {
        for shape in &mut s.shapes.as_mut().expect("must have shapes")[i] {
            let cell = &shape.cells[0];
            if let Ok(c) = tools::lookup_in_image(image, cell.x, cell.y) {
                if c != color {
                    return Err("output shapes have different colors");
                }
            }
        }
    }
    // Recolor shapes.
    for shapes in s.shapes.as_mut().expect("must have shapes") {
        for shape in shapes {
            let mut new_shape = (**shape).clone();
            new_shape.recolor(color);
            *shape = Rc::new(new_shape);
        }
    }
    Ok(())
}

fn draw_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &s.shapes.as_ref().expect("must have shapes")[i];
    let mut new_image = (*s.images[i]).clone();
    for shape in shapes {
        tools::draw_shape(&mut new_image, shape);
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn background_as_normal_color(s: &mut SolverState) -> Res<()> {
    let mut mapping = vec![-1; COLORS.len()];
    for i in 0..COLORS.len() {
        if s.used_colors.contains(&(i as i32)) {
            mapping[i] = i as i32;
        }
    }
    mapping[0] = 11;
    s.apply(|s: &mut SolverState, i: usize| {
        remap_colors(s, i, &mapping);
        Ok(())
    })?;
    s.color_mapping = Some(vec![mapping; s.images.len()]);
    s.used_colors = tools::get_used_colors(&s.images);
    Ok(())
}

fn solve_example_7(task: &Task) -> Res<Vec<Example>> {
    let mut state = SolverState::new(task);
    state.apply(find_colorsets)?;
    use_colorsets_as_shapes(&mut state)?;
    state.apply(remap_colors_by_shapes)?;
    state.apply(find_shapes)?;
    rotate_used_colors(&mut state)?;
    save_shapes(&mut state)?;
    state.apply(filter_shapes_by_color)?;
    save_shapes(&mut state)?;
    load_earlier_shapes(&mut state)?;
    rotate_used_colors(&mut state)?;
    state.apply(filter_shapes_by_color)?;
    state.apply(move_shapes_to_saved_shape)?;
    state.apply(unmap_colors)?;
    Ok(state.get_results())
}

fn solve_example_11(task: &Task) -> Res<Vec<Example>> {
    let mut state = SolverState::new(task);
    state.apply(find_colorsets)?;
    use_colorsets_as_shapes(&mut state)?;
    state.apply(sort_shapes_by_size)?;
    state.apply(remap_colors_by_shapes)?;
    grow_flowers(&mut state)?;
    state.apply(unmap_colors)?;
    Ok(state.get_results())
}

fn solve_example_0(task: &Task) -> Res<Vec<Example>> {
    let mut state = SolverState::new(task);
    state.apply(save_image_as_shape)?;
    resize_image(&mut state)?;
    state.apply(tile_shapes)?;
    state.apply(draw_shape_where_non_empty)?;
    Ok(state.get_results())
}

fn solve_example_1(task: &Task) -> Res<Vec<Example>> {
    let mut state = SolverState::new(task);
    background_as_normal_color(&mut state)?;
    rotate_used_colors(&mut state)?;
    state.apply(find_shapes)?;
    state.apply(filter_shapes_by_color)?;
    state.apply(delete_shapes_touching_border)?;
    recolor_shapes_per_output(&mut state)?;
    state.apply(draw_shapes)?;
    state.apply(unmap_colors)?;
    Ok(state.get_results())
}

type Solver = fn(&Task) -> Res<Vec<Example>>;
pub const SOLVERS: &[Solver] = &[
    solve_example_0,
    solve_example_1,
    solve_example_7,
    solve_example_11,
];
