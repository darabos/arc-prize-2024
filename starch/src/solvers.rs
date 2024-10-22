use crate::tools;
use crate::tools::{Example, Image, Res, Shape, Task, COLORS};
use std::rc::Rc;

macro_rules! err {
    ($msg:expr) => {
        concat!($msg, " at ", file!(), ":", line!())
    };
}

pub fn use_colorsets_as_shapes(s: &mut SolverState) -> Res<()> {
    s.shapes = s.colorsets.clone();
    Ok(())
}

fn order_colors_by_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
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
    let dots = get_firsts(&s.shapes)?;
    // let input_pattern = find_pattern_around(&s.images[..s.task.train.len()], &dots);
    let mut output_pattern = tools::find_pattern_around(&s.output_images, &dots);
    output_pattern.use_relative_colors(&tools::reverse_colors(&s.colors[0]));
    // TODO: Instead of growing each dot, we should filter by the input_pattern.
    s.apply(|s: &mut SolverState, i: usize| {
        let dots = &s.shapes[i][0];
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
    s.saved_shapes.push(s.shapes.clone());
    Ok(())
}

fn load_earlier_shapes(s: &mut SolverState, offset: usize) -> Res<()> {
    if s.saved_shapes.len() - 1 < offset {
        return Err(err!("no saved shapes"));
    }
    s.shapes = s.saved_shapes[s.saved_shapes.len() - 1 - offset].clone();
    Ok(())
}

fn save_shapes_and_load_previous(s: &mut SolverState) -> Res<()> {
    save_shapes(s)?;
    load_earlier_shapes(s, 1)
}

fn save_first_shape_use_the_rest(s: &mut SolverState) -> Res<()> {
    if s.shapes.iter().any(|shapes| shapes.len() < 2) {
        return Err(err!("not enough shapes"));
    }
    let first_shapes: ShapesPerExample = s
        .shapes
        .iter()
        .map(|shapes| vec![shapes[0].clone()])
        .collect();
    s.saved_shapes.push(first_shapes);
    s.shapes = std::mem::take(&mut s.shapes)
        .into_iter()
        .map(|shapes| shapes[1..].to_vec())
        .collect();
    Ok(())
}

fn load_shapes(s: &mut SolverState) -> Res<()> {
    load_earlier_shapes(s, 0)
}

fn load_shapes_except_current_shapes(s: &mut SolverState) -> Res<()> {
    let excluded_shapes = std::mem::take(&mut s.shapes);
    load_shapes(s)?;
    s.shapes = std::mem::take(&mut s.shapes)
        .into_iter()
        .zip(excluded_shapes)
        .map(|(shapes, excluded)| {
            shapes
                .iter()
                .filter(|shape| !excluded.contains(shape))
                .cloned()
                .collect()
        })
        .collect();
    Ok(())
}

fn draw_saved_shapes(s: &mut SolverState) -> Res<()> {
    load_shapes(s)?;
    s.apply(draw_shapes)
}

fn save_whole_image(s: &mut SolverState) -> Res<()> {
    s.saved_images.push(s.images.clone());
    Ok(())
}

type Shapes = Vec<Rc<Shape>>;
type ShapesPerExample = Vec<Shapes>;
type ImagePerExample = Vec<Rc<Image>>;
type ColorList = Vec<i32>;
type ColorListPerExample = Vec<ColorList>;
type LinesPerExample = Vec<tools::Lines>;

/// Tracks information while applying operations on all examples at once.
/// Most fields are vectors storing information for each example.
#[derive(Default, Clone)]
pub struct SolverState {
    pub task: Rc<Task>,
    pub images: ImagePerExample,
    pub saved_images: Vec<ImagePerExample>,
    pub output_images: Vec<Rc<Image>>,
    pub colors: ColorListPerExample,
    pub shapes: ShapesPerExample,
    pub saved_shapes: Vec<ShapesPerExample>,
    pub colorsets: ShapesPerExample,
    pub scale_up: i32,
    pub last_move: tools::Vec2,
    // Lines that go all the way through the image.
    pub horizontal_lines: LinesPerExample,
    pub vertical_lines: LinesPerExample,
}

fn find_shapes(image: &Image) -> Shapes {
    let shapes = tools::find_shapes_in_image(image, &tools::DIRECTIONS4);
    tools::discard_background_shapes_touching_border(image, shapes)
}

impl SolverState {
    pub fn new(task: &Task) -> Self {
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
        let mut state = SolverState {
            task: Rc::new(task.clone()),
            output_images,
            ..Default::default()
        };
        state.init_from_images(images);
        state
    }

    fn init_from_images(&mut self, images: ImagePerExample) {
        self.images = images;
        let all_colors: ColorList = (0..COLORS.len() as i32).collect();
        self.colors = self.images.iter().map(|_| all_colors.clone()).collect();
        self.colorsets = self
            .images
            .iter()
            .map(|image| tools::find_colorsets_in_image(image))
            .collect();
        self.shapes = self.images.iter().map(|image| find_shapes(image)).collect();
        for s in &mut self.shapes {
            s.sort_by_key(|shape| shape.color());
        }
        self.saved_shapes = vec![self.shapes.clone()];
        self.horizontal_lines = self
            .images
            .iter()
            .map(|image| tools::find_horizontal_lines_in_image(image))
            .collect();
        self.vertical_lines = self
            .images
            .iter()
            .map(|image| tools::find_vertical_lines_in_image(image))
            .collect();
        self.apply(order_colors_by_shapes).unwrap();
    }

    fn validate(&self) -> Res<()> {
        if self.images.is_empty() {
            return Err("no images");
        }
        if self.images.len() != self.task.train.len() + self.task.test.len() {
            return Err("wrong number of images");
        }
        if self.images.iter().any(|image| image.is_empty()) {
            return Err("empty image");
        }
        if self.output_images.is_empty() {
            return Err("no output images");
        }
        if self.output_images.len() != self.task.train.len() {
            return Err("wrong number of output images");
        }
        if self.output_images.iter().any(|image| image.is_empty()) {
            return Err("empty output image");
        }
        if self.colors.is_empty() {
            return Err("no colors");
        }
        if self.colors.len() != self.images.len() {
            return Err("wrong number of color lists");
        }
        if self.shapes.is_empty() {
            return Err("no shapes");
        }
        if self.shapes.len() != self.images.len() {
            return Err("wrong number of shape lists");
        }
        if self.colorsets.is_empty() {
            return Err("no colorsets");
        }
        if self.colorsets.len() != self.images.len() {
            return Err("wrong number of colorset lists");
        }
        if self.horizontal_lines.is_empty() {
            return Err("no horizontal lines");
        }
        if self.horizontal_lines.len() != self.images.len() {
            return Err("wrong number of horizontal line lists");
        }
        if self.vertical_lines.is_empty() {
            return Err("no vertical lines");
        }
        if self.vertical_lines.len() != self.images.len() {
            return Err("wrong number of vertical line lists");
        }
        Ok(())
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

    pub fn get_results(&self) -> Vec<Example> {
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
    fn width_and_height_all(&self) -> Res<(i32, i32)> {
        let (w, h) = self.width_and_height(0);
        for i in 1..self.images.len() {
            let (w1, h1) = self.width_and_height(i);
            if w1 != w || h1 != h {
                return Err("images have different sizes");
            }
        }
        Ok((w, h))
    }
    fn output_width_and_height(&self, i: usize) -> (i32, i32) {
        let width = self.output_images[i][0].len() as i32;
        let height = self.output_images[i].len() as i32;
        (width, height)
    }
    fn output_width_and_height_all(&self) -> Res<(i32, i32)> {
        let (w, h) = self.output_width_and_height(0);
        for i in 1..self.output_images.len() {
            let (w1, h1) = self.output_width_and_height(i);
            if w1 != w || h1 != h {
                return Err("output images have different sizes");
            }
        }
        Ok((w, h))
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

    /// An iterator for sub-states that contain only one image and its shapes.
    pub fn state_per_image(&self) -> Vec<SolverState> {
        (0..self.images.len())
            .map(|i| SolverState {
                task: self.task.clone(),
                images: vec![self.images[i].clone()],
                output_images: self.output_images.get(i).cloned().into_iter().collect(),
                colors: vec![self.colors[i].clone()],
                shapes: vec![self.shapes[i].clone()],
                saved_shapes: self
                    .saved_shapes
                    .iter()
                    .map(|s| vec![s[i].clone()])
                    .collect(),
                colorsets: vec![self.colorsets[i].clone()],
                scale_up: self.scale_up,
                last_move: self.last_move,
                ..Default::default()
            })
            .collect()
    }

    pub fn run_steps(&mut self, steps: &[SolverStep]) -> Res<()> {
        for (i, step) in steps.iter().enumerate() {
            self.validate()?;
            match step {
                SolverStep::Each(f) => self.apply(f)?,
                SolverStep::All(f) => f(self)?,
                SolverStep::ForEachShape => {
                    let mut new_images = vec![];
                    for mut state in self.state_per_image() {
                        let shapes = std::mem::take(&mut state.shapes[0]);
                        for shape in shapes {
                            state.shapes = vec![vec![shape.clone()]];
                            state.run_steps(&steps[i + 1..])?;
                        }
                        new_images.push(state.images[0].clone());
                    }
                    self.images = new_images;
                    break;
                }
            }
        }
        Ok(())
    }
}

#[allow(dead_code)]
fn print_shapes(shapes: &ShapesPerExample) {
    for (i, shapes) in shapes.iter().enumerate() {
        println!("{} shapes for example {}", shapes.len(), i);
        for shape in shapes {
            shape.print();
            println!();
        }
    }
}

#[allow(dead_code)]
fn print_shapes_step(s: &mut SolverState) -> Res<()> {
    s.print_shapes();
    Ok(())
}

#[allow(dead_code)]
fn print_images_step(s: &mut SolverState) -> Res<()> {
    s.print_images();
    Ok(())
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
    let shapes = &mut s.shapes[i];
    let color = s.colors[i].get(0).ok_or("no used colors")?;
    s.shapes[i] = shapes
        .iter()
        .filter(|shape| shape.color() == *color)
        .cloned()
        .collect();
    Ok(())
}

fn move_shapes_to_touch_saved_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &s.shapes[i];
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
        return Err(err!("output size must be a multiple of input size"));
    }
    s.scale_up = output_size / input_size;
    s.apply(|s: &mut SolverState, i: usize| {
        s.images[i] = Rc::new(tools::scale_up_image(&s.images[i], s.scale_up as usize));
        Ok(())
    })
}

fn use_image_as_shape(s: &mut SolverState, i: usize) -> Res<()> {
    s.shapes[i] = vec![Rc::new(Shape::from_image(&s.images[i]))];
    Ok(())
}

fn use_image_without_background_as_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let mut shape = Shape::from_image(&s.images[i]);
    shape.discard_color(0);
    s.shapes[i] = vec![shape.into()];
    Ok(())
}

fn tile_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let (current_width, current_height) = s.width_and_height(i);
    let shapes = &mut s.shapes[i];
    let old_width = current_width / s.scale_up;
    let old_height = current_height / s.scale_up;
    for shape in shapes.iter_mut() {
        *shape = shape
            .tile(old_width, current_width, old_height, current_height)
            .into();
    }
    Ok(())
}

fn draw_shape_where_non_empty(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &s.shapes[i];
    let mut new_image = (*s.images[i]).clone();
    for shape in shapes {
        shape.draw_where_non_empty(&mut new_image);
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn delete_shapes_touching_border(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
    *shapes = shapes
        .iter()
        .filter(|shape| !shape.is_touching_border(&s.images[i]))
        .cloned()
        .collect();
    Ok(())
}

fn recolor_shapes_per_output(s: &mut SolverState) -> Res<()> {
    // Get color from first output_image.
    let first_shape = &s.shapes[0].get(0).ok_or("no shapes")?;
    let color = tools::lookup_in_image(
        &s.output_images[0],
        first_shape.cells[0].x,
        first_shape.cells[0].y,
    )?;
    // Fail the operation if any shape in any output has a different color.
    for (i, image) in s.output_images.iter_mut().enumerate() {
        for shape in &mut s.shapes[i] {
            let cell = &shape.cells[0];
            if let Ok(c) = tools::lookup_in_image(image, cell.x, cell.y) {
                if c != color {
                    return Err(err!("output shapes have different colors"));
                }
            }
        }
    }
    // Recolor shapes.
    for shapes in &mut s.shapes {
        for shape in shapes {
            let mut new_shape = (**shape).clone();
            new_shape.recolor(color);
            *shape = new_shape.into();
        }
    }
    Ok(())
}

fn draw_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &s.shapes[i];
    let mut new_image = (*s.images[i]).clone();
    for shape in shapes {
        tools::draw_shape(&mut new_image, shape);
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn order_shapes_by_color(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
    shapes.sort_by_key(|shape| shape.color());
    Ok(())
}

fn order_shapes_by_size_decreasing(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
    shapes.sort_by_key(|shape| -(shape.cells.len() as i32));
    Ok(())
}

fn order_shapes_by_size_increasing(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
    shapes.sort_by_key(|shape| shape.cells.len());
    Ok(())
}

fn find_repeating_pattern(s: &mut SolverState, i: usize) -> Res<()> {
    let (width, height) = s.width_and_height(i);
    let shapes = &mut s.shapes[i];
    let shape = shapes.get(0).ok_or(err!("no shapes"))?;
    let bb = shape.bounding_box();
    if bb.width() < 3 && bb.height() < 3 {
        return Err(err!("shape too small"));
    }
    let mut w = 1;
    let mut h = 1;
    for _ in 0..10 {
        let p = shape.crop(0, 0, w, h);
        let p1 = p.tile(w, bb.right, h, bb.bottom);
        if p1 == **shape {
            let p2 = p.tile(w, width, h, height);
            if p2.cells.is_empty() {
                // This can happen if the image is smaller than the shape.
                return Err(err!("empty pattern"));
            }
            s.shapes[i] = vec![p2.into()];
            return Ok(());
        }
        if w < h && w < width {
            w += 1;
        } else {
            h += 1;
            if height < h {
                break;
            }
        }
    }
    Err(err!("no repeating pattern found"))
}

fn use_output_size(s: &mut SolverState) -> Res<()> {
    let (current_width, current_height) = s.width_and_height(0);
    let (output_width, output_height) = s.output_width_and_height(0);
    if current_width == output_width && current_height == output_height {
        return Err(err!("already correct size"));
    }
    // Make sure all outputs have the same size.
    for i in 1..s.output_images.len() {
        let (w, h) = s.output_width_and_height(i);
        if w != output_width || h != output_height {
            return Err(err!("output images have different sizes"));
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
    let shapes = &mut s.shapes[i];
    let shape = shapes
        .iter()
        .max_by_key(|shape| shape.cells[0])
        .ok_or(err!("no shapes"))?;
    *shapes = vec![shape.clone()];
    Ok(())
}

fn pick_bottom_right_shape_per_color(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
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
        return Err(err!("no shapes"));
    }
    *shapes = new_shapes;
    Ok(())
}

fn allow_diagonals_in_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let mut shapes = tools::find_shapes_in_image(&s.images[i], &tools::DIRECTIONS8);
    shapes = tools::discard_background_shapes_touching_border(&s.images[i], shapes);
    s.shapes[i] = shapes;
    Ok(())
}

fn delete_background_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
    *shapes = shapes
        .iter()
        .filter(|shape| shape.color() != 0)
        .cloned()
        .collect();
    Ok(())
}

fn move_shapes_per_output_shapes(s: &mut SolverState) -> Res<()> {
    let shapes0 = &s.shapes[0];
    let output_shapes0 = find_shapes(&s.output_images[0]);
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
        let shapes = &s.shapes[i];
        let output_shapes = find_shapes(&s.output_images[i]);
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
        let shapes = &mut s.shapes[i];
        *shapes = shapes
            .iter()
            .map(|shape| shape.move_by(offset).into())
            .collect();
    }
    Ok(())
}

fn move_shapes_per_output(s: &mut SolverState) -> Res<()> {
    let shapes = &s.shapes;
    let outputs = &s.output_images;
    for distance in 1..5 {
        for direction in tools::DIRECTIONS8 {
            let mut correct = true;
            let offset = distance * direction;
            for i in 0..outputs.len() {
                for shape in &shapes[i] {
                    if !shape.matches_image_when_moved_by(&outputs[i], offset) {
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

/// Moves the saved shape in one of the 8 directions as far as possible while still covering the
/// current shape. Works with a single saved shape and current shape.
fn move_saved_shape_to_cover_current_shape_max(s: &mut SolverState, i: usize) -> Res<()> {
    let saved_shapes = &s.saved_shapes.last().ok_or(err!("no saved shapes"))?[i];
    let current_shape = &s.shapes[i].get(0).ok_or(err!("no current shape"))?;
    let saved_shape = saved_shapes.get(0).ok_or(err!("no saved shape"))?;
    let mut moved: Shape = (**saved_shape).clone();
    for distance in (1..10).rev() {
        for direction in tools::DIRECTIONS8 {
            moved.move_by_mut(distance * direction);
            if moved.covers(current_shape) {
                s.shapes[i] = vec![moved.into()];
                s.last_move = distance * direction;
                return Ok(());
            }
            moved.restore_from(&saved_shape);
        }
    }
    Err(err!("no move found"))
}

/// Draws the shape in its current location, then moves it again and draws it again,
/// until it leaves the image.
fn repeat_last_move_and_draw(s: &mut SolverState, i: usize) -> Res<()> {
    if s.last_move == tools::Vec2::ZERO {
        return Err(err!("no last move"));
    }
    let mut shapes: Vec<Shape> = s.shapes[i].iter().map(|shape| (**shape).clone()).collect();
    let mut new_image = (*s.images[i]).clone();
    for _ in 0..10 {
        for shape in &mut shapes {
            tools::draw_shape(&mut new_image, &shape);
            shape.move_by_mut(s.last_move);
        }
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn recolor_saved_shapes_to_current_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let saved_shapes = &s.saved_shapes.last().ok_or(err!("no saved shapes"))?[i];
    let current_shape = &s.shapes[i].get(0).ok_or(err!("no current shape"))?;
    let color = current_shape.color();
    let mut new_saved_shapes = vec![];
    for saved_shape in saved_shapes {
        let mut new_shape = (**saved_shape).clone();
        new_shape.recolor(color);
        new_saved_shapes.push(new_shape.into());
    }
    let len = s.saved_shapes.len();
    s.saved_shapes[len - 1][i] = new_saved_shapes;
    Ok(())
}

fn split_into_two_images(s: &mut SolverState) -> Res<()> {
    let (width, height) = s.width_and_height_all()?;
    let (output_width, output_height) = s.output_width_and_height_all()?;
    if width == output_width * 2 + 1 {
        let mut to_save = vec![];
        for image in &mut s.images {
            let left_image = Rc::new(tools::crop_image(image, 0, 0, width / 2, height));
            let right_image = Rc::new(tools::crop_image(
                image,
                width / 2 + 1,
                0,
                width / 2,
                height,
            ));
            *image = left_image;
            to_save.push(right_image);
        }
        s.saved_images.push(to_save);
        return Ok(());
    } else if height == output_height * 2 + 1 {
        let mut to_save = vec![];
        for image in &mut s.images {
            let top_image = Rc::new(tools::crop_image(image, 0, 0, width, height / 2));
            let bottom_image = Rc::new(tools::crop_image(
                image,
                0,
                height / 2 + 1,
                width,
                height / 2,
            ));
            *image = top_image;
            to_save.push(bottom_image);
        }
        s.saved_images.push(to_save);
        return Ok(());
    }
    Err(err!("no split found"))
}

fn boolean_with_saved_image_and(s: &mut SolverState, i: usize) -> Res<()> {
    boolean_with_saved_image_function(s, i, |a, b| if a == 0 { 0 } else { b })
}
fn boolean_with_saved_image_or(s: &mut SolverState, i: usize) -> Res<()> {
    boolean_with_saved_image_function(s, i, |a, b| if a == 0 { b } else { a })
}
fn boolean_with_saved_image_xor(s: &mut SolverState, i: usize) -> Res<()> {
    boolean_with_saved_image_function(s, i, |a, b| {
        if a == 0 {
            b
        } else if b == 0 {
            a
        } else {
            0
        }
    })
}

fn boolean_with_saved_image_function(
    s: &mut SolverState,
    i: usize,
    func: fn(i32, i32) -> i32,
) -> Res<()> {
    let saved_image = &s.saved_images.last().ok_or(err!("no saved images"))?[i];
    let mut new_image: Image = (*s.images[i]).clone();
    for i in 0..new_image.len() {
        for j in 0..new_image[0].len() {
            let a = new_image[i][j];
            let b = saved_image[i][j];
            new_image[i][j] = func(a, b);
        }
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn recolor_image_per_output(s: &mut SolverState) -> Res<()> {
    let used_colors = tools::get_used_colors(&s.output_images);
    if used_colors.len() != 1 {
        return Err(err!("output images have different colors"));
    }
    let color = used_colors[0];
    for image in &mut s.images {
        let new_image = image
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&c| if c == 0 { 0 } else { color })
                    .collect()
            })
            .collect();
        *image = Rc::new(new_image);
    }
    Ok(())
}

fn find_matching_offset(
    shape: &ShapesPerExample,
    images: &ImagePerExample,
    direction: tools::Vec2,
    start_pos: tools::Vec2,
    min_offset: i32,
    max_offset: i32,
) -> Res<i32> {
    for distance in min_offset..=max_offset {
        if are_shapes_present_at(shape, images, start_pos + distance * direction) {
            return Ok(distance);
        }
    }
    Err(err!("no match found"))
}

fn are_shapes_present_at(
    shapes: &ShapesPerExample,
    images: &ImagePerExample,
    pos: tools::Vec2,
) -> bool {
    for i in 0..images.len() {
        let image = &images[i];
        for shape in &shapes[i] {
            if !shape.matches_image_when_moved_by(image, pos) {
                return false;
            }
        }
    }
    true
}

fn repeat_shapes_on_lattice_per_output(s: &mut SolverState) -> Res<()> {
    // Make sure the shapes are not tiny.
    for shapes_per in &s.shapes {
        let total_cells: usize = shapes_per.iter().map(|s| s.cells.len()).sum();
        if total_cells < 4 {
            return Err(err!("shapes too small"));
        }
    }
    // Find a lattice. This is a periodic pattern described by two parameters,
    // the horizontal period (a single number) and the vertical offset (a Vec2).
    let horizontal_period = find_matching_offset(
        &s.shapes,
        &s.output_images,
        tools::RIGHT,
        tools::Vec2::ZERO,
        1,
        10,
    )?;
    for vertical_y in 1..3 {
        let vertical_x = find_matching_offset(
            &s.shapes,
            &s.output_images,
            tools::RIGHT,
            vertical_y * tools::DOWN,
            -2,
            2,
        )?;
        if are_shapes_present_at(
            &s.shapes,
            &s.output_images,
            vertical_y * tools::DOWN + (vertical_x + horizontal_period) * tools::RIGHT,
        ) {
            return repeat_shapes_on_lattice(s, horizontal_period, vertical_x, vertical_y);
        }
        if are_shapes_present_at(
            &s.shapes,
            &s.output_images,
            vertical_y * tools::DOWN + (vertical_x - horizontal_period) * tools::RIGHT,
        ) {
            return repeat_shapes_on_lattice(s, horizontal_period, vertical_x, vertical_y);
        }
    }
    Err(err!("no match found"))
}

fn repeat_shapes_on_lattice(
    s: &mut SolverState,
    horizontal_period: i32,
    vertical_x: i32,
    vertical_y: i32,
) -> Res<()> {
    for i in 0..s.images.len() {
        let image = &s.images[i];
        let shapes = &s.shapes[i];
        let mut new_image = (**image).clone();
        for shape in shapes {
            for rep_x in -5..=5 {
                for rep_y in -5..=5 {
                    let pos = tools::Vec2 {
                        x: rep_x * horizontal_period + rep_y * vertical_x,
                        y: rep_y * vertical_y,
                    };
                    tools::draw_shape_at(&mut new_image, shape, pos);
                }
            }
        }
        s.images[i] = Rc::new(new_image);
    }
    Ok(())
}

/// Deletes the lines. Instead of erasing them, we completely remove them from the image.
fn remove_horizontal_lines(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &*s.images[i];
    let mut new_image = vec![];
    let lines = &s.horizontal_lines[i];
    let mut line_i = 0;
    let (_width, height) = s.width_and_height(i);
    for y in 0..height {
        if line_i < lines.len() && y == lines[line_i].pos {
            line_i += 1;
        } else {
            new_image.push(image[y as usize].clone());
        }
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn remove_vertical_lines(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &*s.images[i];
    let mut new_image = vec![];
    let lines = &s.vertical_lines[i];
    let (width, height) = s.width_and_height(i);
    for y in 0..height {
        let mut row = vec![];
        let mut line_i = 0;
        for x in 0..width {
            if line_i < lines.len() && x == lines[line_i].pos {
                line_i += 1;
            } else {
                row.push(image[y as usize][x as usize]);
            }
        }
        new_image.push(row);
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn remove_grid(s: &mut SolverState, i: usize) -> Res<()> {
    remove_horizontal_lines(s, i)?;
    remove_vertical_lines(s, i)?;
    // Keep the horizontal and vertical lines so we can restore the grid later.
    let horizontal_lines = std::mem::take(&mut s.horizontal_lines);
    let vertical_lines = std::mem::take(&mut s.vertical_lines);
    let images = std::mem::take(&mut s.images);
    s.init_from_images(images);
    s.horizontal_lines = horizontal_lines;
    s.vertical_lines = vertical_lines;
    Ok(())
}

fn insert_horizontal_lines(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &*s.images[i];
    let mut new_image = vec![];
    let lines = &s.horizontal_lines[i];
    let mut line_i = 0;
    let (width, height) = s.width_and_height(i);
    for y in 0..height {
        while line_i < lines.len() && new_image.len() == lines[line_i].pos as usize {
            new_image.push(vec![lines[line_i].color; width as usize]);
            line_i += 1;
        }
        new_image.push(image[y as usize].clone());
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn insert_vertical_lines(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &*s.images[i];
    let mut new_image = vec![];
    let lines = &s.vertical_lines[i];
    let (width, height) = s.width_and_height(i);
    for y in 0..height {
        let mut row = vec![];
        let mut line_i = 0;
        for x in 0..width {
            while line_i < lines.len() && row.len() == lines[line_i].pos as usize {
                row.push(lines[line_i].color);
                line_i += 1;
            }
            row.push(image[y as usize][x as usize]);
        }
        new_image.push(row);
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn restore_grid(s: &mut SolverState, i: usize) -> Res<()> {
    insert_horizontal_lines(s, i)?;
    insert_vertical_lines(s, i)?;
    Ok(())
}

fn connect_aligned_pixels_in_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let mut new_image = (*s.images[i]).clone();
    for shape in &s.shapes[i] {
        let bb = shape.bounding_box();
        let shape_as_image = shape.as_image();
        for cell in &shape.cells {
            for dir in tools::DIRECTIONS4 {
                for distance in 1..10 {
                    let image_pos = *cell + distance * dir;
                    let shape_pos = image_pos - bb.top_left();
                    if tools::lookup_in_image(&shape_as_image, shape_pos.x, shape_pos.y)
                        .unwrap_or(0)
                        != 0
                    {
                        for d in 1..distance {
                            let pos = *cell + d * dir;
                            new_image[pos.y as usize][pos.x as usize] = cell.color;
                        }
                    }
                }
            }
        }
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

pub enum SolverStep {
    Each(fn(&mut SolverState, usize) -> Res<()>),
    All(fn(&mut SolverState) -> Res<()>),
    ForEachShape,
}
use SolverStep::*;
pub const SOLVERS: &[&[SolverStep]] = &[
    &[
        // 0
        Each(use_image_as_shape),
        All(scale_up_image),
        Each(tile_shapes),
        Each(draw_shape_where_non_empty),
    ],
    &[
        // 1
        Each(filter_shapes_by_color),
        All(recolor_shapes_per_output),
        Each(draw_shapes),
    ],
    &[
        // 2
        All(use_output_size),
        All(use_colorsets_as_shapes),
        Each(find_repeating_pattern),
        All(recolor_shapes_per_output),
        Each(draw_shapes),
    ],
    &[
        // 3
        Each(pick_bottom_right_shape_per_color),
        All(load_shapes_except_current_shapes),
        All(move_shapes_per_output),
    ],
    &[
        // 4
        Each(allow_diagonals_in_shapes),
        Each(delete_background_shapes),
        Each(order_shapes_by_size_decreasing),
        All(save_first_shape_use_the_rest),
        ForEachShape,
        Each(recolor_saved_shapes_to_current_shape),
        Each(move_saved_shape_to_cover_current_shape_max),
        Each(repeat_last_move_and_draw),
    ],
    &[
        // 5
        All(split_into_two_images),
        Each(boolean_with_saved_image_and),
        All(recolor_image_per_output),
    ],
    &[
        // 6
        Each(use_image_without_background_as_shape),
        All(repeat_shapes_on_lattice_per_output),
    ],
    &[
        // 7
        Each(use_next_color),
        Each(filter_shapes_by_color),
        All(save_shapes_and_load_previous),
        Each(use_previous_color),
        Each(filter_shapes_by_color),
        Each(move_shapes_to_touch_saved_shape),
    ],
    &[
        // 8
        Each(remove_grid),
        All(use_colorsets_as_shapes),
        Each(connect_aligned_pixels_in_shapes),
        Each(restore_grid),
    ],
    &[
        // 11
        All(use_colorsets_as_shapes),
        Each(order_shapes_by_size_increasing),
        Each(order_colors_by_shapes),
        All(grow_flowers),
    ],
    &[
        // 71
        All(split_into_two_images),
        Each(boolean_with_saved_image_xor),
        All(recolor_image_per_output),
    ],
    &[
        // ???
        All(split_into_two_images),
        Each(boolean_with_saved_image_or),
        All(recolor_image_per_output),
    ],
];
