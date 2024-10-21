use crate::tools;
use crate::tools::{Example, Image, Res, Shape, Task, COLORS};
use std::rc::Rc;

macro_rules! err {
    ($msg:expr) => {
        concat!($msg, " at ", file!(), ":", line!())
    };
}

fn init_shape_vec(n: usize, vec: &mut Option<Vec<Vec<Rc<Shape>>>>) {
    if vec.is_none() {
        *vec = Some((0..n).map(|_| vec![]).collect());
    }
}

pub fn find_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    init_shape_vec(s.images.len(), &mut s.shapes);
    s.shapes.as_mut().unwrap()[i] = tools::find_shapes_in_image(&s.images[i], true);
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
    let shapes = &mut s.shapes.as_mut().ok_or(err!("must have shapes"))?[i];
    tools::sort_shapes_by_size(shapes);
    Ok(())
}

fn order_colors_by_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes.as_mut().ok_or(err!("must have shapes"))?[i];
    let mut already_used = vec![false; COLORS.len()];
    let mut next_color = 0;
    for shape in shapes.iter() {
        if !already_used[shape.color() as usize] {
            s.colors[i][next_color] = shape.color();
            already_used[shape.color() as usize] = true;
            next_color += 1;
        }
    }
    // Unused colors at the end.
    for color in 0..COLORS.len() {
        if !already_used[color] {
            s.colors[i][next_color] = color as i32;
            next_color += 1;
        }
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
    let mut output_pattern = tools::find_pattern_around(&s.output_images, &dots);
    output_pattern.use_relative_colors(&tools::reverse_colors(&s.colors[0]));
    // TODO: Instead of growing each dot, we should filter by the input_pattern.
    s.apply(|s: &mut SolverState, i: usize| {
        let shapes = &s.shapes.as_ref().ok_or("must have shapes")?;
        let dots = &shapes[i][0];
        let mut new_image = (*s.images[i]).clone();
        for dot in dots.cells.iter() {
            tools::draw_shape_with_relative_colors_at(
                &mut new_image,
                &output_pattern,
                &s.colors[i],
                &dot.pos(),
            );
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

fn load_earlier_shapes(s: &mut SolverState, offset: usize) -> Res<()> {
    if s.saved_shapes.len() - 1 < offset {
        return Err("no saved shapes");
    }
    s.shapes = Some(
        s.saved_shapes
            .get(s.saved_shapes.len() - 1 - offset)
            .unwrap()
            .clone(),
    );
    Ok(())
}

fn save_shapes_and_load_previous(s: &mut SolverState) -> Res<()> {
    save_shapes(s)?;
    load_earlier_shapes(s, 1)
}

fn load_shapes(s: &mut SolverState) -> Res<()> {
    load_earlier_shapes(s, 0)
}

fn load_shapes_except_current_shapes(s: &mut SolverState) -> Res<()> {
    let excluded_shapes = s.shapes.take().ok_or("must have shapes")?;
    load_shapes(s)?;
    s.shapes = Some(
        s.shapes
            .take()
            .ok_or("must have shapes")?
            .into_iter()
            .zip(excluded_shapes)
            .map(|(shapes, excluded)| {
                shapes
                    .iter()
                    .filter(|shape| !excluded.contains(shape))
                    .cloned()
                    .collect()
            })
            .collect(),
    );
    Ok(())
}

fn save_whole_image(s: &mut SolverState) -> Res<()> {
    s.saved_images.push(s.images.clone());
    Ok(())
}

type Shapes = Vec<Rc<Shape>>;
type ShapesPerExample = Vec<Shapes>;
type ImagesPerExample = Vec<Rc<Image>>;
type ColorList = Vec<i32>;
type ColorListPerExample = Vec<ColorList>;

/// Tracks information while applying operations on all examples at once.
/// Most fields are vectors storing information for each example.
#[derive(Default)]
pub struct SolverState {
    pub task: Rc<Task>,
    pub images: ImagesPerExample,
    pub saved_images: Vec<ImagesPerExample>,
    pub output_images: Vec<Rc<Image>>,
    pub colors: ColorListPerExample,
    pub shapes: Option<ShapesPerExample>,
    pub saved_shapes: Vec<ShapesPerExample>,
    pub colorsets: Option<ShapesPerExample>,
    pub scale_up: i32,
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
        let all_colors: ColorList = (0..COLORS.len() as i32).collect();
        let colors = images.iter().map(|_| all_colors.clone()).collect();
        SolverState {
            task: Rc::new(task.clone()),
            images,
            output_images,
            colors,
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
    fn width_and_height(&self, i: usize) -> (i32, i32) {
        let width = self.images[i][0].len() as i32;
        let height = self.images[i].len() as i32;
        (width, height)
    }
    fn output_width_and_height(&self, i: usize) -> (i32, i32) {
        let width = self.output_images[i][0].len() as i32;
        let height = self.output_images[i].len() as i32;
        (width, height)
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

    #[allow(dead_code)]
    fn print_colors(&self) {
        for colors in &self.colors {
            for color in colors {
                tools::print_color(*color);
            }
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

fn use_next_color(s: &mut SolverState, i: usize) -> Res<()> {
    let first_color = s.colors[i][0];
    let n = COLORS.len();
    for j in 0..n - 1 {
        s.colors[i][j] = s.colors[i][j + 1];
    }
    s.colors[i][n - 1] = first_color;
    Ok(())
}

fn use_previous_color(s: &mut SolverState, i: usize) -> Res<()> {
    let last_color = s.colors[i][COLORS.len() - 1];
    let n = COLORS.len();
    for j in (1..n).rev() {
        s.colors[i][j] = s.colors[i][j - 1];
    }
    s.colors[i][0] = last_color;
    Ok(())
}

fn filter_shapes_by_color(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes.as_mut().ok_or("must have shapes")?[i];
    let color = s.colors[i].get(0).ok_or("no used colors")?;
    s.shapes.as_mut().unwrap()[i] = shapes
        .iter()
        .filter(|shape| shape.color() == *color)
        .cloned()
        .collect();
    Ok(())
}

fn move_shapes_to_saved_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &s.shapes.as_ref().ok_or("must have shapes")?[i];
    let saved_shapes = &s.saved_shapes.last().ok_or("must have saved shapes")?[i];
    let saved_shape = saved_shapes.get(0).ok_or("saved shapes empty")?;
    let mut new_image = (*s.images[i]).clone();
    for shape in shapes {
        new_image = tools::move_shape_to_shape_in_image(&new_image, &shape, &saved_shape)?;
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn scale_up_image(s: &mut SolverState) -> Res<()> {
    // Find ratio from looking at example outputs.
    let output_size = s.output_images[0].len() as i32;
    let input_size = s.images[0].len() as i32;
    if output_size % input_size != 0 {
        return Err("output size must be a multiple of input size");
    }
    s.scale_up = output_size / input_size;
    s.apply(|s: &mut SolverState, i: usize| {
        s.images[i] = Rc::new(tools::scale_up_image(&s.images[i], s.scale_up as usize));
        Ok(())
    })
}

fn save_image_as_shape(s: &mut SolverState, i: usize) -> Res<()> {
    init_shape_vec(s.images.len(), &mut s.shapes);
    s.shapes.as_mut().unwrap()[i] = vec![Rc::new(Shape::from_image(&s.images[i]))];
    Ok(())
}

fn tile_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let (current_width, current_height) = s.width_and_height(i);
    let shapes = &mut s.shapes.as_mut().ok_or("must have shapes")?[i];
    let old_width = current_width / s.scale_up;
    let old_height = current_height / s.scale_up;
    for shape in shapes.iter_mut() {
        let mut new_shape = (**shape).clone();
        new_shape.tile(old_width, current_width, old_height, current_height);
        *shape = Rc::new(new_shape);
    }
    Ok(())
}

fn draw_shape_where_non_empty(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &s.shapes.as_ref().ok_or("must have shapes")?[i];
    let mut new_image = (*s.images[i]).clone();
    for shape in shapes {
        shape.draw_where_non_empty(&mut new_image);
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn delete_shapes_touching_border(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes.as_mut().ok_or("must have shapes")?[i];
    *shapes = shapes
        .iter()
        .filter(|shape| !shape.is_touching_border(&s.images[i]))
        .cloned()
        .collect();
    Ok(())
}

fn recolor_shapes_per_output(s: &mut SolverState) -> Res<()> {
    // Get color from first output_image.
    let first_shape = &s.shapes.as_ref().ok_or("must have shapes")?[0]
        .get(0)
        .ok_or("no shapes")?;
    let color = tools::lookup_in_image(
        &s.output_images[0],
        first_shape.cells[0].x,
        first_shape.cells[0].y,
    )?;
    // Fail the operation if any shape in any output has a different color.
    for (i, image) in s.output_images.iter_mut().enumerate() {
        for shape in &mut s.shapes.as_mut().ok_or("must have shapes")?[i] {
            let cell = &shape.cells[0];
            if let Ok(c) = tools::lookup_in_image(image, cell.x, cell.y) {
                if c != color {
                    return Err("output shapes have different colors");
                }
            }
        }
    }
    // Recolor shapes.
    for shapes in s.shapes.as_mut().ok_or("must have shapes")? {
        for shape in shapes {
            let mut new_shape = (**shape).clone();
            new_shape.recolor(color);
            *shape = Rc::new(new_shape);
        }
    }
    Ok(())
}

fn draw_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &s.shapes.as_ref().ok_or("must have shapes")?[i];
    let mut new_image = (*s.images[i]).clone();
    for shape in shapes {
        tools::draw_shape(&mut new_image, shape);
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn find_shapes_include_background(s: &mut SolverState, i: usize) -> Res<()> {
    init_shape_vec(s.images.len(), &mut s.shapes);
    s.shapes.as_mut().unwrap()[i] = tools::find_shapes_in_image(&s.images[i], false);
    Ok(())
}

fn order_shapes_by_color(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes.as_mut().ok_or("must have shapes")?[i];
    shapes.sort_by_key(|shape| shape.color());
    Ok(())
}

fn find_repeating_pattern(s: &mut SolverState, i: usize) -> Res<()> {
    let (width, height) = s.width_and_height(i);
    let shapes = &mut s.shapes.as_mut().ok_or(err!("must have shapes"))?[i];
    let shape = shapes.get(0).ok_or(err!("no shapes"))?;
    let bb = shape.bounding_box();
    let mut w = 1;
    let mut h = 1;
    for _ in 0..10 {
        let p = shape.crop(0, 0, w, h);
        let p1 = p.tile(w, bb.right, h, bb.bottom);
        if p1 == **shape {
            let p2 = p.tile(w, width, h, height);
            s.shapes.as_mut().unwrap()[i] = vec![p2.into()];
            return Ok(());
        }
        if w < h {
            w += 1;
        } else {
            h += 1;
        }
    }
    Err("no repeating pattern found")
}

fn use_output_size(s: &mut SolverState) -> Res<()> {
    let (current_width, current_height) = s.width_and_height(0);
    let (output_width, output_height) = s.output_width_and_height(0);
    if current_width == output_width && current_height == output_height {
        return Err("already correct size");
    }
    // Make sure all outputs have the same size.
    for i in 1..s.output_images.len() {
        let (w, h) = s.output_width_and_height(i);
        if w != output_width || h != output_height {
            return Err("output images have different sizes");
        }
    }
    s.apply(|s: &mut SolverState, i: usize| {
        s.images[i] = Rc::new(tools::resize_canvas(
            &s.images[i],
            output_width as usize,
            output_height as usize,
        ));
        Ok(())
    })
}

fn pick_bottom_right_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes.as_mut().ok_or(err!("must have shapes"))?[i];
    let shape = shapes
        .iter()
        .max_by_key(|shape| shape.cells[0])
        .ok_or(err!("no shapes"))?;
    *shapes = vec![shape.clone()];
    Ok(())
}

fn pick_bottom_right_shape_per_color(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes.as_mut().ok_or(err!("must have shapes"))?[i];
    let mut new_shapes = vec![];
    for color in &s.colors[i] {
        let shape = shapes
            .iter()
            .filter(|shape| shape.color() == *color)
            .max_by_key(|shape| shape.bounding_box().bottom_right());
        if let Some(shape) = shape {
            new_shapes.push(shape.clone());
        }
    }
    if new_shapes.is_empty() {
        return Err("no shapes");
    }
    *shapes = new_shapes;
    Ok(())
}

fn move_shapes_per_output_shapes(s: &mut SolverState) -> Res<()> {
    let shapes0 = &s.shapes.as_ref().ok_or(err!("must have shapes"))?[0];
    let output_shapes0 = tools::find_shapes_in_image(&s.output_images[0], true);
    let relative_output_shapes0: Shapes = output_shapes0
        .iter()
        .map(|shape| shape.to_relative_pos().into())
        .collect();
    // Figure out the offset.
    let in00 = shapes0.get(0).ok_or(err!("no shape"))?;
    let out0_index = in00
        .to_relative_pos()
        .find_matching_shape_index(&relative_output_shapes0)
        .ok_or(err!("no match"))?;
    let out0 = &output_shapes0[out0_index];
    let offset = in00.cells[0] - out0.cells[0];
    // Confirm that this offset is correct for all shapes in all examples.
    for i in 0..s.output_images.len() {
        let shapes = &s.shapes.as_ref().ok_or(err!("must have shapes"))?[i];
        let output_shapes = tools::find_shapes_in_image(&s.output_images[i], true);
        let relative_output_shapes: Shapes = output_shapes
            .iter()
            .map(|shape| shape.to_relative_pos().into())
            .collect();
        for shape in shapes {
            let out_index = shape
                .to_relative_pos()
                .find_matching_shape_index(&relative_output_shapes)
                .ok_or(err!("no match"))?;
            let out = &output_shapes[out_index];
            if shape.cells[0] - out.cells[0] != offset {
                return Err(err!("offsets don't match"));
            }
        }
    }
    // Move shapes.
    for i in 0..s.images.len() {
        let shapes = &mut s.shapes.as_mut().ok_or(err!("must have shapes"))?[i];
        *shapes = shapes
            .iter()
            .map(|shape| shape.move_by(offset).into())
            .collect();
    }
    Ok(())
}

fn move_shapes_per_output(s: &mut SolverState) -> Res<()> {
    let shapes = &s.shapes.as_ref().ok_or(err!("must have shapes"))?;
    let outputs = &s.output_images;
    for distance in 1..5 {
        for direction in tools::DIRECTIONS {
            let mut correct = true;
            let offset = distance * direction;
            for i in 0..outputs.len() {
                for shape in &shapes[i] {
                    if !shape.match_image_when_moved_by(&outputs[i], offset) {
                        correct = false;
                        break;
                    }
                }
            }
            if correct {
                for i in 0..s.images.len() {
                    let mut new_image = (*s.images[i]).clone();
                    for shape in &shapes[i] {
                        tools::remove_shape(&mut new_image, shape);
                        tools::draw_shape_at(&mut new_image, shape, offset);
                    }
                    s.images[i] = Rc::new(new_image);
                }
                return Ok(());
            }
        }
    }
    Err(err!("no match found"))
}

fn solve_example_7(task: &Task) -> Res<Vec<Example>> {
    let mut state = SolverState::new(task);
    state.apply(find_shapes)?;
    state.apply(order_shapes_by_color)?;
    state.apply(order_colors_by_shapes)?;
    state.apply(use_next_color)?;
    save_shapes(&mut state)?;
    state.apply(filter_shapes_by_color)?;
    save_shapes_and_load_previous(&mut state)?;
    state.apply(use_previous_color)?;
    state.apply(filter_shapes_by_color)?;
    state.apply(move_shapes_to_saved_shape)?;
    Ok(state.get_results())
}

fn solve_example_11(task: &Task) -> Res<Vec<Example>> {
    let mut state = SolverState::new(task);
    state.apply(find_colorsets)?;
    use_colorsets_as_shapes(&mut state)?;
    state.apply(sort_shapes_by_size)?;
    state.apply(order_colors_by_shapes)?;
    grow_flowers(&mut state)?;
    Ok(state.get_results())
}

fn solve_example_0(task: &Task) -> Res<Vec<Example>> {
    let mut state = SolverState::new(task);
    state.apply(save_image_as_shape)?;
    scale_up_image(&mut state)?;
    state.apply(tile_shapes)?;
    state.apply(draw_shape_where_non_empty)?;
    Ok(state.get_results())
}

fn solve_example_1(task: &Task) -> Res<Vec<Example>> {
    let mut state = SolverState::new(task);
    state.apply(find_shapes_include_background)?;
    state.apply(filter_shapes_by_color)?;
    state.apply(delete_shapes_touching_border)?;
    recolor_shapes_per_output(&mut state)?;
    state.apply(draw_shapes)?;
    Ok(state.get_results())
}

fn solve_example_2(task: &Task) -> Res<Vec<Example>> {
    let mut state = SolverState::new(task);
    use_output_size(&mut state)?;
    state.apply(find_colorsets)?;
    use_colorsets_as_shapes(&mut state)?;
    state.apply(find_repeating_pattern)?;
    recolor_shapes_per_output(&mut state)?;
    state.apply(draw_shapes)?;
    Ok(state.get_results())
}

fn solve_example_3(task: &Task) -> Res<Vec<Example>> {
    let mut state = SolverState::new(task);
    state.apply(find_shapes)?;
    save_shapes(&mut state)?;
    state.apply(pick_bottom_right_shape_per_color)?;
    load_shapes_except_current_shapes(&mut state)?;
    move_shapes_per_output(&mut state)?;
    Ok(state.get_results())
}

type Solver = fn(&Task) -> Res<Vec<Example>>;
pub const SOLVERS: &[Solver] = &[
    solve_example_0,
    solve_example_1,
    solve_example_2,
    solve_example_3,
    solve_example_7,
    solve_example_11,
];
