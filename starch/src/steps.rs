use crate::solvers::*;
use crate::tools;
use crate::tools::{Image, Res, Shape, Vec2, COLORS};
use std::rc::Rc;

macro_rules! err {
    ($msg:expr) => {
        concat!($msg, " at ", file!(), ":", line!())
    };
}

/// Returns an error if the given Vec is empty.
macro_rules! must_not_be_empty {
    ($e:expr) => {
        if $e.is_empty() {
            return Err(err!("empty"));
        }
    };
}
/// Returns an error if any of the Vecs is empty.
macro_rules! must_all_be_non_empty {
    ($e:expr) => {
        if $e.iter().any(|e| e.is_empty()) {
            return Err(err!("empty"));
        }
    };
}

pub fn use_colorsets_as_shapes(s: &mut SolverState) -> Res<()> {
    s.shapes = s.colorsets.clone();
    Ok(())
}

pub fn order_colors_by_shapes(s: &mut SolverState, i: usize) -> Res<()> {
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
pub fn get_firsts<T>(vec: &Vec<Vec<T>>) -> Res<Vec<&T>> {
    let mut firsts = vec![];
    for e in vec {
        if e.is_empty() {
            return Err("empty list");
        }
        firsts.push(&e[0]);
    }
    Ok(firsts)
}

pub fn grow_flowers_square(s: &mut SolverState) -> Res<()> {
    grow_flowers(s, tools::find_pattern_in_square)
}
pub fn grow_flowers_horizontally(s: &mut SolverState) -> Res<()> {
    grow_flowers(s, tools::find_pattern_horizontally)
}
pub fn grow_flowers_vertically(s: &mut SolverState) -> Res<()> {
    grow_flowers(s, tools::find_pattern_vertically)
}

pub fn grow_flowers<F>(s: &mut SolverState, func: F) -> Res<()>
where
    F: Fn(&[Rc<Image>], &[&Rc<Shape>]) -> Res<Shape>,
{
    must_not_be_empty!(&s.output_images);
    // It's okay for some images to not have dots.
    let indexes: Vec<usize> = s
        .shapes
        .iter()
        .enumerate()
        .filter_map(|(i, shapes)| {
            let dots: usize = shapes.iter().map(|shape| shape.pixels.len()).sum();
            if i >= s.output_images.len() || dots == 0 || dots > 5 {
                None
            } else {
                Some(i)
            }
        })
        .collect();
    if indexes.is_empty() {
        return Err(err!("no dots"));
    }
    let dots: Vec<&Rc<Shape>> = indexes.iter().map(|&i| &s.shapes[i][0]).collect();
    let output_images: Vec<Rc<Image>> = indexes
        .iter()
        .map(|&i| &s.output_images[i])
        .cloned()
        .collect();
    match func(&output_images, &dots) {
        Ok(pattern) => s.apply(|s: &mut SolverState, i: usize| {
            let mut new_image = (*s.images[i]).clone();
            for dots in &s.shapes[i] {
                for dot in dots.cells() {
                    new_image.draw_shape_at(&pattern, dot.pos());
                }
            }
            s.images[i] = Rc::new(new_image);
            Ok(())
        }),
        Err(_) => Ok(()), // We grew a nil pattern.
    }
}

pub fn save_shapes(s: &mut SolverState) -> Res<()> {
    s.saved_shapes.push(s.shapes.clone());
    Ok(())
}

pub fn load_earlier_shapes(s: &mut SolverState, offset: usize) -> Res<()> {
    if s.saved_shapes.len() < offset + 1 {
        return Err(err!("no saved shapes"));
    }
    s.shapes = s.saved_shapes[s.saved_shapes.len() - 1 - offset].clone();
    Ok(())
}

pub fn save_shapes_and_load_previous(s: &mut SolverState) -> Res<()> {
    save_shapes(s)?;
    load_earlier_shapes(s, 1)
}

pub fn save_first_shape_use_the_rest(s: &mut SolverState) -> Res<()> {
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

pub fn take_first_shape_save_the_rest(s: &mut SolverState) -> Res<()> {
    if s.shapes.iter().any(|shapes| shapes.len() < 2) {
        return Err(err!("not enough shapes"));
    }
    let first_shapes: ShapesPerExample = s
        .shapes
        .iter()
        .map(|shapes| vec![shapes[0].clone()])
        .collect();
    let the_rest = std::mem::take(&mut s.shapes)
        .into_iter()
        .map(|shapes| shapes[1..].to_vec())
        .collect();
    s.shapes = first_shapes;
    s.saved_shapes.push(the_rest);
    Ok(())
}

pub fn load_shapes(s: &mut SolverState) -> Res<()> {
    load_earlier_shapes(s, 0)
}

pub fn load_shapes_except_current_shapes(s: &mut SolverState) -> Res<()> {
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
    must_all_be_non_empty!(&s.shapes);
    Ok(())
}

pub fn draw_saved_shapes(s: &mut SolverState) -> Res<()> {
    load_shapes(s)?;
    s.apply(draw_shapes)
}

pub fn save_whole_image(s: &mut SolverState) -> Res<()> {
    s.saved_images.push(s.images.clone());
    Ok(())
}

pub fn draw_saved_image(s: &mut SolverState) -> Res<()> {
    let saved_images = s.saved_images.pop().ok_or("no saved images")?;
    for i in 0..s.images.len() {
        let mut new_image = (*s.images[i]).clone();
        new_image.draw_image_at(&*saved_images[i], Vec2::ZERO);
        s.images[i] = new_image.into();
    }
    Ok(())
}

#[allow(dead_code)]
pub fn print_shapes_step(s: &mut SolverState) -> Res<()> {
    s.print_shapes();
    Ok(())
}

#[allow(dead_code)]
pub fn print_images_step(s: &mut SolverState) -> Res<()> {
    println!("Images after {:?}", s.steps);
    s.print_images();
    Ok(())
}

#[allow(dead_code)]
pub fn print_colors_step(s: &mut SolverState) -> Res<()> {
    s.print_colors();
    Ok(())
}

pub fn substates_for_each_image(s: &mut SolverState) -> Res<()> {
    if s.is_substate {
        return Err(err!("already split into substates"));
    }
    s.substates = Some(s.substate_per_image());
    Ok(())
}

pub fn substates_for_each_shape(s: &mut SolverState) -> Res<()> {
    if s.is_substate {
        return Err(err!("already split into substates"));
    }
    if s.shapes.iter().any(|shapes| shapes.len() > 10) {
        return Err(err!("too many shapes"));
    }
    s.substates = Some(
        s.substate_per_image()
            .into_iter()
            .map(|mut s| {
                let shapes = std::mem::take(&mut s.state.shapes[0]);
                shapes.into_iter().map(move |shape| {
                    let mut state = s.state.clone();
                    state.shapes = vec![vec![shape.clone()]];
                    SubState {
                        state,
                        image_indexes: s.image_indexes.clone(),
                    }
                })
            })
            .flatten()
            .collect(),
    );
    Ok(())
}

pub fn substates_for_each_color(s: &mut SolverState) -> Res<()> {
    let used_colors = tools::get_used_colors(&s.images);
    if used_colors.len() < 2 {
        return Err(err!("not enough colors"));
    }
    s.substates = Some(
        used_colors
            .into_iter()
            .map(|color| {
                let mut substate = s.substate();
                substate.state.colors =
                    vec![tools::add_remaining_colors(&vec![color]); s.images.len()];
                substate
            })
            .collect(),
    );
    Ok(())
}

pub fn use_next_color(s: &mut SolverState, i: usize) -> Res<()> {
    let first_color = s.colors[i][0];
    let n = COLORS.len();
    for j in 0..n - 1 {
        s.colors[i][j] = s.colors[i][j + 1];
    }
    s.colors[i][n - 1] = first_color;
    Ok(())
}

pub fn use_previous_color(s: &mut SolverState, i: usize) -> Res<()> {
    let last_color = s.colors[i][COLORS.len() - 1];
    let n = COLORS.len();
    for j in (1..n).rev() {
        s.colors[i][j] = s.colors[i][j - 1];
    }
    s.colors[i][0] = last_color;
    Ok(())
}

pub fn filter_shapes_by_color(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
    let color = s.colors[i].get(0).ok_or("no used colors")?;
    s.shapes[i] = shapes
        .iter()
        .filter(|shape| shape.color() == *color)
        .cloned()
        .collect();
    Ok(())
}

pub fn move_shapes_to_touch_saved_shape(s: &mut SolverState, i: usize) -> Res<()> {
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

/// Scales up the image to match the output image.
pub fn scale_up_image(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(&s.output_images);
    // Find ratio from looking at example outputs.
    let output_width = s.output_images[0].width as i32;
    let input_width = s.images[0].width as i32;
    if output_width % input_width != 0 {
        return Err(err!("output width must be a multiple of input width"));
    }
    let scale_x = output_width / input_width;
    let output_height = s.output_images[0].height as i32;
    let input_height = s.images[0].height as i32;
    if output_height % input_height != 0 {
        return Err(err!("output height must be a multiple of input height"));
    }
    let scale_y = output_height / input_height;
    s.scale_up = Vec2 {
        x: scale_x,
        y: scale_y,
    };
    // Check that the same ratio applies to all images.
    for (image, output) in s.images.iter().zip(&s.output_images) {
        if image.width as i32 * scale_x != output.width as i32
            || image.height as i32 * scale_y != output.height as i32
        {
            return Err(err!("scaling is not consistent"));
        }
    }
    s.apply(|s: &mut SolverState, i: usize| {
        s.images[i] = Rc::new(tools::scale_up_image(&s.images[i], s.scale_up));
        Ok(())
    })
}

/// Scales up the image to match the output image after adding a grid stored in horizontal_lines and
/// vertical_lines.
pub fn scale_up_image_add_grid(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(&s.output_images);
    let lines = &s.saved_lines;
    must_not_be_empty!(lines);
    let lines = &lines[lines.len() - 1];
    let num_h = lines[0].horizontal.len() as i32;
    let num_v = lines[0].vertical.len() as i32;
    if num_h + num_v == 0 {
        return Err(err!("no grid"));
    }
    if lines.iter().any(|lines| {
        lines.horizontal.len() != num_h as usize || lines.vertical.len() != num_v as usize
    }) {
        return Err(err!("lines have different lengths"));
    }
    // Find ratio from looking at example outputs.
    let (output_width, output_height) = s.output_width_and_height_all()?;
    let (width, height) = s.width_and_height_all()?;
    if (output_width - num_v) % width != 0 {
        return Err(err!("output width must be a multiple of input width"));
    }
    if (output_height - num_h) % height != 0 {
        return Err(err!("output width must be a multiple of input width"));
    }
    s.scale_up = Vec2 {
        x: (output_width - num_v) / width,
        y: (output_height - num_h) / height,
    };
    s.apply(|s: &mut SolverState, i: usize| {
        s.images[i] = Rc::new(tools::scale_up_image(&s.images[i], s.scale_up));
        Ok(())
    })?;
    restore_grid(s)
}

/// Tiles the image to match the output image.
pub fn tile_image(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(&s.output_images);
    // Find ratio from looking at example outputs.
    let output_width = s.output_images[0].width;
    let input_width = s.images[0].width;
    if output_width % input_width != 0 {
        return Err(err!("output width must be a multiple of input width"));
    }
    let scale_x = output_width / input_width;
    let output_height = s.output_images[0].height;
    let input_height = s.images[0].height;
    if output_height % input_height != 0 {
        return Err(err!("output height must be a multiple of input height"));
    }
    let scale_y = output_height / input_height;
    // Check that the same ratio applies to all images.
    for (image, output) in s.images.iter().zip(&s.output_images) {
        if image.width * scale_x != output.width || image.height * scale_y != output.height {
            return Err(err!("scaling is not consistent"));
        }
    }
    s.apply(|s: &mut SolverState, i: usize| {
        s.images[i] = Rc::new(tools::tile_image(&s.images[i], scale_x, scale_y));
        Ok(())
    })
}

/// Tiles the image to match the output image.
pub fn tile_image_add_grid(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(&s.output_images);
    // Find ratio from looking at example outputs.
    let lines = &s.saved_lines;
    must_not_be_empty!(lines);
    let lines = &lines[lines.len() - 1];
    let num_h = lines[0].horizontal.len() as i32;
    let num_v = lines[0].vertical.len() as i32;
    if num_h + num_v == 0 {
        return Err(err!("no grid"));
    }
    if lines.iter().any(|lines| {
        lines.horizontal.len() != num_h as usize || lines.vertical.len() != num_v as usize
    }) {
        return Err(err!("lines have different lengths"));
    }
    let (output_width, output_height) = s.output_width_and_height_all()?;
    let (width, height) = s.width_and_height_all()?;
    if (output_width - num_v) % width != 0 {
        return Err(err!("output width must be a multiple of input width"));
    }
    if (output_height - num_h) % height != 0 {
        return Err(err!("output width must be a multiple of input width"));
    }
    let scale_x = (output_width - num_v) / width;
    let scale_y = (output_height - num_h) / height;
    s.apply(|s: &mut SolverState, i: usize| {
        s.images[i] = Rc::new(tools::tile_image(
            &s.images[i],
            scale_x as usize,
            scale_y as usize,
        ));
        Ok(())
    })?;
    restore_grid(s)
}

pub fn use_image_as_shape(s: &mut SolverState, i: usize) -> Res<()> {
    s.shapes[i] = vec![Rc::new(Shape::from_image(&s.images[i]))];
    Ok(())
}

pub fn use_image_without_background_as_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let shape = Shape::from_image(&s.images[i]).discard_color(0)?;
    s.shapes[i] = vec![shape.into()];
    Ok(())
}

pub fn tile_shapes_after_scale_up(s: &mut SolverState, i: usize) -> Res<()> {
    if s.scale_up.x <= 1 && s.scale_up.y <= 1 {
        return Err(err!("not scaled up"));
    }
    let (current_width, current_height) = s.width_and_height(i);
    let shapes = &mut s.shapes[i];
    let old_width = current_width / s.scale_up.x;
    let old_height = current_height / s.scale_up.y;
    for shape in shapes.iter_mut() {
        *shape = shape
            .tile(old_width, current_width, old_height, current_height)?
            .into();
    }
    Ok(())
}

pub fn draw_shape_where_non_empty(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &s.shapes[i];
    let mut new_image = (*s.images[i]).clone();
    for shape in shapes {
        shape.draw_where_non_empty(&mut new_image);
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

pub fn discard_shapes_touching_border(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
    *shapes = shapes
        .iter()
        .filter(|shape| !shape.is_touching_border(&s.images[i]))
        .cloned()
        .collect();
    must_not_be_empty!(shapes);
    Ok(())
}

pub fn recolor_shapes_per_output(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(&s.output_images);
    // Get colors from first output_image.
    let shapes = &s.shapes[0];
    let colors: Res<Vec<i32>> = shapes
        .iter()
        .map(|shape| s.output_images[0].get(shape.cell0().x, shape.cell0().y))
        .collect();
    let mut colors = colors?;
    if colors.is_empty() {
        return Err(err!("no colors"));
    }
    let all_same_color = colors.iter().all(|&c| c == colors[0]);
    // Fail the operation if any shape in any output has a different color.
    for (i, image) in s.output_images.iter_mut().enumerate() {
        for (j, shape) in &mut s.shapes[i].iter().enumerate() {
            let cell = shape.cell0();
            if let Ok(c) = image.get(cell.x, cell.y) {
                if all_same_color {
                    if c != colors[0] {
                        return Err(err!("output shapes have different colors"));
                    }
                } else {
                    while j >= colors.len() {
                        colors.push(c);
                    }
                    if c != colors[j] {
                        return Err(err!("output shapes have different colors"));
                    }
                }
            }
        }
    }
    // Recolor shapes.
    for shapes in &mut s.shapes {
        for (j, shape) in shapes.iter_mut().enumerate() {
            let color = if all_same_color {
                colors[0]
            } else {
                colors[j % colors.len()]
            };
            *shape = shape.recolor(color).into();
        }
    }
    Ok(())
}

pub fn draw_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &s.shapes[i];
    let mut new_image = (*s.images[i]).clone();
    for shape in shapes {
        new_image.draw_shape(shape);
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

pub fn order_shapes_by_color(s: &mut SolverState, i: usize) -> Res<()> {
    s.shapes[i].sort_by_key(|shape| shape.color());
    if i < s.output_shapes.len() {
        s.output_shapes[i].sort_by_key(|shape| shape.color());
    }
    Ok(())
}

pub fn order_shapes_by_weight_decreasing(s: &mut SolverState, i: usize) -> Res<()> {
    s.shapes[i].sort_by_key(|shape| -(shape.pixels.len() as i32));
    if i < s.output_shapes.len() {
        s.output_shapes[i].sort_by_key(|shape| -(shape.pixels.len() as i32));
    }
    Ok(())
}

pub fn order_shapes_by_weight_increasing(s: &mut SolverState, i: usize) -> Res<()> {
    s.shapes[i].sort_by_key(|shape| shape.pixels.len());
    if i < s.output_shapes.len() {
        s.output_shapes[i].sort_by_key(|shape| shape.pixels.len());
    }
    Ok(())
}

pub fn order_shapes_by_bb_size_decreasing(s: &mut SolverState, i: usize) -> Res<()> {
    s.shapes[i].sort_by_key(|shape| -(shape.bb.area()));
    if i < s.output_shapes.len() {
        s.output_shapes[i].sort_by_key(|shape| -(shape.bb.area() as i32));
    }
    Ok(())
}

pub fn order_shapes_by_bb_size_increasing(s: &mut SolverState, i: usize) -> Res<()> {
    s.shapes[i].sort_by_key(|shape| shape.bb.area());
    if i < s.output_shapes.len() {
        s.output_shapes[i].sort_by_key(|shape| shape.bb.area());
    }
    Ok(())
}

pub fn order_shapes_from_left_to_right(s: &mut SolverState, i: usize) -> Res<()> {
    s.shapes[i].sort_by_key(|shape| shape.bb.left);
    if i < s.output_shapes.len() {
        s.output_shapes[i].sort_by_key(|shape| shape.bb.left);
    }
    Ok(())
}
pub fn order_shapes_from_top_to_bottom(s: &mut SolverState, i: usize) -> Res<()> {
    s.shapes[i].sort_by_key(|shape| shape.bb.top);
    if i < s.output_shapes.len() {
        s.output_shapes[i].sort_by_key(|shape| shape.bb.top);
    }
    Ok(())
}

pub fn find_repeating_pattern_in_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let (width, height) = s.width_and_height(i);
    let shapes = &mut s.shapes[i];
    let shape = shapes.get(0).ok_or(err!("no shapes"))?;
    if shape.bb.width() < 3 && shape.bb.height() < 3 {
        return Err(err!("shape too small"));
    }
    let mut w = 1;
    let mut h = 1;
    for _ in 0..10 {
        let pattern = tile_shape(shape, w, h, width, height);
        if let Ok(pattern) = pattern {
            s.shapes[i] = vec![pattern.into()];
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

pub fn tile_shape(
    shape: &Shape,
    shape_w: i32,
    shape_h: i32,
    image_w: i32,
    image_h: i32,
) -> Res<Shape> {
    let p = shape.crop(0, 0, shape_w, shape_h)?;
    let p1 = p.tile(shape_w, shape.bb.right, shape_h, shape.bb.bottom)?;
    if p1 != *shape {
        return Err(err!("not matching"));
    }
    p.tile(shape_w, image_w, shape_h, image_h)
}

pub fn use_output_size(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(&s.output_images);
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
        s.images[i] = Rc::new(Image::new(output_width as usize, output_height as usize));
        Ok(())
    })
}

pub fn pick_bottom_right_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
    let shape = shapes
        .iter()
        .max_by_key(|shape| shape.bb.bottom_right())
        .ok_or(err!("no shapes"))?;
    *shapes = vec![shape.clone()];
    Ok(())
}

pub fn pick_bottom_right_shape_per_color(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
    let mut new_shapes = vec![];
    for color in &s.colors[i] {
        let shape = shapes
            .iter()
            .filter(|shape| shape.color() == *color)
            .max_by_key(|shape| shape.bb.bottom_right());
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

pub fn allow_diagonals_in_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let mut shapes = tools::find_shapes_in_image(&s.images[i], &Vec2::DIRECTIONS8);
    shapes = tools::discard_background_shapes_touching_border(&s.images[i], shapes);
    s.shapes[i] = shapes;
    Ok(())
}
pub fn allow_diagonals_in_multicolor_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let mut shapes = tools::find_multicolor_shapes_in_image(&s.images[i], &Vec2::DIRECTIONS8);
    shapes = tools::discard_background_shapes_touching_border(&s.images[i], shapes);
    s.shapes[i] = shapes;
    Ok(())
}

pub fn discard_background_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
    *shapes = shapes
        .iter()
        .filter(|shape| shape.color() != 0)
        .cloned()
        .collect();
    must_not_be_empty!(shapes);
    Ok(())
}

pub fn move_shapes_per_output_shapes(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(&s.output_shapes);
    let shapes0 = &s.shapes[0];
    let output_shapes0 = &s.output_shapes[0];
    // Figure out the offset.
    let in00 = shapes0.get(0).ok_or(err!("no shape"))?;
    let out0_index = in00
        .find_matching_shape_index(&output_shapes0)
        .ok_or(err!("no match"))?;
    let out0 = &output_shapes0[out0_index];
    let offset = in00.bb.top_left() - out0.bb.top_left();
    // Confirm that this offset is correct for all shapes in all examples.
    for i in 0..s.output_images.len() {
        let shapes = &s.shapes[i];
        let output_shapes = &s.output_shapes[i];
        for shape in shapes {
            let out_index = shape
                .find_matching_shape_index(&output_shapes)
                .ok_or(err!("no match"))?;
            let out = &output_shapes[out_index];
            if shape.bb.top_left() - out.bb.top_left() != offset {
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

pub fn move_shapes_per_output(s: &mut SolverState) -> Res<()> {
    let shapes = &s.shapes;
    let outputs = &s.output_images;
    for distance in 1..5 {
        for direction in Vec2::DIRECTIONS8 {
            let mut correct = true;
            let offset = distance * direction;
            'images: for i in 0..outputs.len() {
                for shape in &shapes[i] {
                    if !shape.matches_image_when_moved_by(&outputs[i], offset) {
                        correct = false;
                        break 'images;
                    }
                }
            }
            if correct {
                for i in 0..s.images.len() {
                    let mut new_image = (*s.images[i]).clone();
                    for shape in &shapes[i] {
                        new_image.erase_shape(shape);
                        new_image.draw_shape_at(shape, offset);
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
pub fn move_saved_shape_to_cover_current_shape_max(s: &mut SolverState, i: usize) -> Res<()> {
    let saved_shapes = &s.saved_shapes.last().ok_or(err!("no saved shapes"))?[i];
    let current_shape = &s.shapes[i].get(0).ok_or(err!("no current shape"))?;
    let saved_shape = saved_shapes.get(0).ok_or(err!("no saved shape"))?;
    let mut moved: Shape = (**saved_shape).clone();
    for distance in (1..10).rev() {
        for direction in Vec2::DIRECTIONS8 {
            moved.move_to_mut(saved_shape.bb.top_left() + distance * direction);
            if moved.covers(current_shape) {
                s.shapes[i] = vec![moved.into()];
                s.last_move[i] = distance * direction;
                return Ok(());
            }
        }
    }
    Err(err!("no move found"))
}

/// Draws the shape in its current location, then moves it again and draws it again,
/// until it leaves the image.
pub fn repeat_last_move_and_draw(s: &mut SolverState, i: usize) -> Res<()> {
    if s.last_move[i] == Vec2::ZERO {
        return Err(err!("no last move"));
    }
    let mut shapes: Vec<Shape> = s.shapes[i].iter().map(|shape| (**shape).clone()).collect();
    let mut new_image = (*s.images[i]).clone();
    for _ in 0..10 {
        for shape in &mut shapes {
            new_image.draw_shape(&shape);
            shape.move_by_mut(s.last_move[i]);
        }
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

pub fn recolor_saved_shapes_to_current_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let saved_shapes = &s.saved_shapes.last().ok_or(err!("no saved shapes"))?[i];
    let current_shape = &s.shapes[i].get(0).ok_or(err!("no current shape"))?;
    let color = current_shape.color();
    let mut new_saved_shapes = vec![];
    for saved_shape in saved_shapes {
        let new_shape = saved_shape.recolor(color);
        new_saved_shapes.push(new_shape.into());
    }
    let len = s.saved_shapes.len();
    s.saved_shapes[len - 1][i] = new_saved_shapes;
    Ok(())
}

pub fn split_into_two_images(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(&s.output_images);
    let (width, height) = s.width_and_height_all()?;
    let (output_width, output_height) = s.output_width_and_height_all()?;
    if width == output_width * 2 + 1 {
        let mut to_save = vec![];
        for image in &mut s.images {
            let left_image = Rc::new(image.crop(0, 0, width / 2, height));
            let right_image = Rc::new(image.crop(width / 2 + 1, 0, width / 2, height));
            *image = left_image;
            to_save.push(right_image);
        }
        s.saved_images.push(to_save);
        return Ok(());
    } else if height == output_height * 2 + 1 {
        let mut to_save = vec![];
        for image in &mut s.images {
            let top_image = Rc::new(image.crop(0, 0, width, height / 2));
            let bottom_image = Rc::new(image.crop(0, height / 2 + 1, width, height / 2));
            *image = top_image;
            to_save.push(bottom_image);
        }
        s.saved_images.push(to_save);
        return Ok(());
    }
    Err(err!("no split found"))
}

pub fn boolean_with_saved_image_and(s: &mut SolverState, i: usize) -> Res<()> {
    boolean_with_saved_image_function(s, i, |a, b| if a == 0 { 0 } else { b })
}
pub fn boolean_with_saved_image_nand(s: &mut SolverState, i: usize) -> Res<()> {
    boolean_with_saved_image_function(s, i, |a, b| if a == 0 && b == 0 { 1 } else { 0 })
}
pub fn boolean_with_saved_image_or(s: &mut SolverState, i: usize) -> Res<()> {
    boolean_with_saved_image_function(s, i, |a, b| if b == 0 { a } else { b })
}
pub fn boolean_with_saved_image_nor(s: &mut SolverState, i: usize) -> Res<()> {
    boolean_with_saved_image_function(s, i, |a, b| if a == 0 || b == 0 { 1 } else { 0 })
}
pub fn boolean_with_saved_image_xor(s: &mut SolverState, i: usize) -> Res<()> {
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

pub fn boolean_with_saved_image_function(
    s: &mut SolverState,
    i: usize,
    func: fn(i32, i32) -> i32,
) -> Res<()> {
    let saved_image = &s.saved_images.last().ok_or(err!("no saved images"))?[i];
    let (width, height) = s.width_and_height(i);
    let (saved_width, saved_height) = tools::width_and_height(&saved_image);
    if width != saved_width || height != saved_height {
        return Err(err!("images have different sizes"));
    }
    let mut new_image: Image = (*s.images[i]).clone();
    for y in 0..new_image.height {
        for x in 0..new_image.width {
            let a = new_image[(x, y)];
            let b = saved_image[(x, y)];
            new_image[(x, y)] = func(a, b);
        }
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

pub fn recolor_image_per_output(s: &mut SolverState) -> Res<()> {
    let used_colors = tools::get_used_colors(&s.output_images);
    if used_colors.len() != 1 {
        return Err(err!("output images have different colors"));
    }
    let color = used_colors[0];
    for image in &mut s.images {
        let mut new_image: Image = (**image).clone();
        new_image.update(|_x, _y, c| if c == 0 { 0 } else { color });
        *image = new_image.into();
    }
    Ok(())
}

pub fn find_matching_offset(
    shapes: &ShapesPerExample,
    images: &ImagePerExample,
    direction: Vec2,
    start_pos: Vec2,
    min_offset: i32,
    max_offset: i32,
) -> Res<i32> {
    for distance in min_offset..=max_offset {
        if are_shapes_present_at(shapes, images, start_pos + distance * direction) {
            return Ok(distance);
        }
    }
    Err(err!("no match found"))
}

pub fn are_shapes_present_at(
    shapes: &ShapesPerExample,
    images: &ImagePerExample,
    pos: Vec2,
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

pub fn repeat_shapes_on_lattice_per_output(s: &mut SolverState) -> Res<()> {
    repeat_shapes_on_lattice_per_reference(s, &s.output_images.clone())
}

pub fn repeat_shapes_on_lattice_per_image(s: &mut SolverState) -> Res<()> {
    repeat_shapes_on_lattice_per_reference(s, &s.images.clone())
}

pub fn repeat_shapes_on_lattice_per_reference(
    s: &mut SolverState,
    references: &ImagePerExample,
) -> Res<()> {
    // Make sure the shapes are not tiny.
    for shapes_per in &s.shapes {
        let total_cells: usize = shapes_per.iter().map(|s| s.pixels.len()).sum();
        if total_cells < 4 {
            return Err(err!("shapes too small"));
        }
    }
    // Find a lattice. This is a periodic pattern described by two parameters,
    // the horizontal period (a single number) and the vertical offset (a Vec2).
    let horizontal_period =
        find_matching_offset(&s.shapes, references, Vec2::RIGHT, Vec2::ZERO, 1, 10)?;
    for vertical_y in 1..10 {
        for m in [1, -1].iter() {
            let vertical_y = vertical_y * *m;
            if let Ok(vertical_x) = find_matching_offset(
                &s.shapes,
                references,
                Vec2::RIGHT,
                vertical_y * Vec2::DOWN,
                -2,
                2,
            ) {
                if are_shapes_present_at(
                    &s.shapes,
                    references,
                    vertical_y * Vec2::DOWN + (vertical_x + horizontal_period) * Vec2::RIGHT,
                ) {
                    return repeat_shapes_on_lattice(s, horizontal_period, vertical_x, vertical_y);
                }
                if are_shapes_present_at(
                    &s.shapes,
                    references,
                    vertical_y * Vec2::DOWN + (vertical_x - horizontal_period) * Vec2::RIGHT,
                ) {
                    return repeat_shapes_on_lattice(s, horizontal_period, vertical_x, vertical_y);
                }
            }
        }
    }
    Err(err!("no match found"))
}

pub fn repeat_shapes_on_lattice(
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
                    let pos = Vec2 {
                        x: rep_x * horizontal_period + rep_y * vertical_x,
                        y: rep_y * vertical_y,
                    };
                    new_image.draw_shape_at(shape, pos);
                }
            }
        }
        s.images[i] = Rc::new(new_image);
    }
    Ok(())
}

pub fn repeat_shapes_vertically(s: &mut SolverState, i: usize) -> Res<()> {
    let (width, height) = s.width_and_height(i);
    let shapes = &mut s.shapes[i];
    for shape in shapes {
        let bb = shape.bb;
        *shape = shape.tile(0, width, bb.height(), height)?.into();
    }
    Ok(())
}

pub fn repeat_shapes_horizontally(s: &mut SolverState, i: usize) -> Res<()> {
    let (width, height) = s.width_and_height(i);
    let shapes = &mut s.shapes[i];
    for shape in shapes {
        let bb = shape.bb;
        *shape = shape.tile(bb.width(), width, 0, height)?.into();
    }
    Ok(())
}

/// Deletes the lines. We don't erase them -- we completely remove them from the image.
pub fn remove_horizontal_lines(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &*s.images[i];
    let mut new_image = vec![];
    let lines = &s.lines[i].horizontal;
    let mut line_i = 0;
    let (_width, height) = s.width_and_height(i);
    'y: for y in 0..height {
        while line_i < lines.len() {
            let li = &lines[line_i];
            if li.pos + li.width as i32 <= y {
                // We are past the line. Look at the next line.
                line_i += 1;
            } else if li.pos <= y {
                // We are inside the line. Drop this row.
                continue 'y;
            } else {
                // We are before the line. Keep this row.
                break;
            }
        }
        let mut row = vec![];
        for x in 0..image.width {
            row.push(image[(x, y as usize)]);
        }
        new_image.push(row);
    }
    if new_image.len() == 0 {
        return Err(err!("nothing left"));
    }
    s.images[i] = Rc::new(Image::from_vecvec(new_image));
    Ok(())
}

pub fn remove_vertical_lines(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &*s.images[i];
    let mut new_image = vec![];
    let lines = &s.lines[i].vertical;
    let (width, height) = s.width_and_height(i);
    for y in 0..height {
        let mut row = vec![];
        let mut line_i = 0;
        'x: for x in 0..width {
            while line_i < lines.len() {
                let li = &lines[line_i];
                if li.pos + li.width as i32 <= x {
                    // We are past the line. Look at the next line.
                    line_i += 1;
                } else if li.pos <= x {
                    // We are inside the line. Drop this pixel.
                    continue 'x;
                } else {
                    // We are before the line. Keep this pixel.
                    break;
                }
            }
            row.push(image[(x as usize, y as usize)]);
        }
        if row.len() == 0 {
            return Err(err!("nothing left"));
        }
        new_image.push(row);
    }
    s.images[i] = Rc::new(Image::from_vecvec(new_image));
    Ok(())
}

pub fn remove_grid(s: &mut SolverState) -> Res<()> {
    // TODO: Kinda slow. Not all lines are a grid. Error out if the lines don't make a grid.
    s.apply(|s: &mut SolverState, i: usize| {
        let (width, height) = s.width_and_height(i);
        if width < 3 && height < 3 {
            return Err(err!("image too small"));
        }
        if s.lines[i].horizontal.is_empty() && s.lines[i].vertical.is_empty() {
            return Err(err!("no grid"));
        }
        remove_horizontal_lines(s, i)?;
        remove_vertical_lines(s, i)?;
        Ok(())
    })?;
    // Keep the horizontal and vertical lines so we can restore the grid later.
    let lines = std::mem::take(&mut s.lines);
    let images = std::mem::take(&mut s.images);
    s.init_from_images(images);
    s.saved_lines.push(lines);
    Ok(())
}

pub fn insert_horizontal_lines(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &*s.images[i];
    let mut new_image = vec![];
    let lines = &s.saved_lines;
    let lines = &lines[lines.len() - 1][i].horizontal;
    let mut line_i = 0;
    let (width, height) = s.width_and_height(i);
    for y in 0..=height {
        while line_i < lines.len() && new_image.len() == lines[line_i].pos as usize {
            for _ in 0..lines[line_i].width {
                new_image.push(vec![lines[line_i].color; width as usize]);
            }
            line_i += 1;
        }
        if y < height {
            let mut row = vec![];
            for x in 0..width {
                row.push(image[(x as usize, y as usize)]);
            }
            new_image.push(row);
        }
    }
    s.images[i] = Rc::new(Image::from_vecvec(new_image));
    Ok(())
}

pub fn insert_vertical_lines(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &*s.images[i];
    let mut new_image = vec![];
    let lines = &s.saved_lines;
    let lines = &lines[lines.len() - 1][i].vertical;
    let (width, height) = s.width_and_height(i);
    for y in 0..height {
        let mut row = vec![];
        let mut line_i = 0;
        for x in 0..=width {
            while line_i < lines.len() && row.len() == lines[line_i].pos as usize {
                for _ in 0..lines[line_i].width {
                    row.push(lines[line_i].color);
                }
                line_i += 1;
            }
            if x < width {
                row.push(image[(x as usize, y as usize)]);
            }
        }
        new_image.push(row);
    }
    s.images[i] = Rc::new(Image::from_vecvec(new_image));
    Ok(())
}

pub fn restore_grid(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(&s.saved_lines);
    s.apply(|s: &mut SolverState, i: usize| {
        insert_horizontal_lines(s, i)?;
        insert_vertical_lines(s, i)?;
        Ok(())
    })?;
    s.saved_lines.pop();
    s.init_from_current_images();
    Ok(())
}

pub fn connect_aligned_pixels_in_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let mut new_image = (*s.images[i]).clone();
    for shape in &s.shapes[i] {
        let bb = shape.bb;
        let shape_as_image = shape.as_image();
        for cell in shape.cells() {
            for dir in Vec2::DIRECTIONS4 {
                for distance in 1..10 {
                    let image_pos = cell + distance * dir;
                    let shape_pos = image_pos - bb.top_left();
                    if shape_as_image.get_or(shape_pos.x, shape_pos.y, 0) != 0 {
                        for d in 1..distance {
                            let pos = cell + d * dir;
                            let _ = new_image.set(pos.x, pos.y, cell.color);
                        }
                    }
                }
            }
        }
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

pub fn select_grid_cell_most_filled_in(s: &mut SolverState) -> Res<()> {
    select_grid_cell_max_by(s, |image| tools::count_non_zero_pixels(image) as i32)
}

pub fn select_grid_cell_least_filled_in(s: &mut SolverState) -> Res<()> {
    select_grid_cell_max_by(s, |image| -(tools::count_non_zero_pixels(image) as i32))
}

/// Adds up the width of all lines.
pub fn total_width(lines: &tools::Lines) -> usize {
    lines.iter().map(|l| l.width).sum()
}

pub fn select_grid_cell_max_by<F>(s: &mut SolverState, score_func: F) -> Res<()>
where
    F: Fn(&Image) -> i32,
{
    select_grid_cell(s, |grid_cells: &[Image]| {
        let mut best_index = 0;
        let mut best_score = std::i32::MIN;
        for (index, c) in grid_cells.iter().enumerate() {
            let score = score_func(c);
            if score > best_score {
                best_score = score;
                best_index = index;
            }
        }
        Ok(best_index)
    })
}

pub fn select_grid_cell_outlier_by_color(s: &mut SolverState) -> Res<()> {
    select_grid_cell(s, |grid_cells: &[Image]| {
        let mut users = vec![vec![]; COLORS.len()];
        for (index, c) in grid_cells.iter().enumerate() {
            let mut counts = vec![0; COLORS.len()];
            tools::count_colors_in_image(c, &mut counts);
            for (color, &count) in counts.iter().enumerate() {
                if count > 0 {
                    users[color].push(index);
                }
            }
        }
        for users in users {
            if users.len() == 1 {
                return Ok(users[0]);
            }
        }
        Err(err!("no outlier found"))
    })
}

pub fn select_grid_cell<F>(s: &mut SolverState, select_func: F) -> Res<()>
where
    F: Fn(&[Image]) -> Res<usize>,
{
    s.apply(|s: &mut SolverState, i: usize| {
        let lines = &s.lines[i];
        if lines.horizontal.len() + lines.vertical.len() == 0 {
            return Err(err!("no grid"));
        }
        let (width, height) = s.width_and_height(i);
        if (width as usize) < (total_width(&lines.horizontal) + lines.horizontal.len() + 1)
            || (height as usize) < (total_width(&lines.vertical) + lines.vertical.len() + 1)
        {
            return Err(err!("image too small"));
        }
        let image = &s.images[i];
        let mut grid_cells = tools::grid_cut_image(image, &lines);
        if grid_cells.is_empty() {
            return Err(err!("nothing left after cutting"));
        }
        let selected = select_func(&grid_cells)?;
        s.images[i] = grid_cells.swap_remove(selected).into();
        Ok(())
    })?;
    // Keep the horizontal and vertical lines so we can restore the grid later.
    let lines = std::mem::take(&mut s.lines);
    let images = std::mem::take(&mut s.images);
    s.init_from_images(images);
    s.saved_lines.push(lines);
    Ok(())
}

pub fn rotate_to_landscape_cw(s: &mut SolverState) -> Res<()> {
    rotate_to_landscape(s, tools::Rotation::CW)
}

pub fn rotate_to_landscape_ccw(s: &mut SolverState) -> Res<()> {
    rotate_to_landscape(s, tools::Rotation::CCW)
}

pub fn rotate_to_landscape(s: &mut SolverState, direction: tools::Rotation) -> Res<()> {
    let needs_rotating: Vec<bool> = (0..s.images.len())
        .map(|i| {
            let (w, h) = s.width_and_height(i);
            h > w
        })
        .collect();
    if needs_rotating.iter().all(|&rotate| !rotate) {
        return Err(err!("already landscape"));
    }
    let new_images = rotate_some_images(&s.images, &needs_rotating, direction);
    let new_output_images = rotate_some_images(&s.output_images, &needs_rotating, direction);
    s.output_images = new_output_images;
    s.init_from_images(new_images);
    s.add_finishing_step(move |s: &mut SolverState| {
        for (image, rotate) in s.images.iter_mut().zip(&needs_rotating) {
            if *rotate {
                *image = tools::rotate_image(image, direction.opposite()).into();
            }
        }
        Ok(())
    });
    Ok(())
}

pub fn rotate_some_images(
    images: &ImagePerExample,
    needs_rotating: &Vec<bool>,
    direction: tools::Rotation,
) -> ImagePerExample {
    images
        .iter()
        .zip(needs_rotating)
        .map(|(image, rotate)| {
            if *rotate {
                tools::rotate_image(image, direction).into()
            } else {
                (*image).clone()
            }
        })
        .collect()
}

pub fn refresh_from_image(s: &mut SolverState) -> Res<()> {
    let images = s.images.clone();
    s.init_from_images(images);
    Ok(())
}

pub fn allow_background_color_shapes(s: &mut SolverState) -> Res<()> {
    if s.shapes == s.shapes_including_background
        && s.output_shapes == s.output_shapes_including_background
    {
        return Err(err!("already including background"));
    }
    s.shapes = s.shapes_including_background.clone();
    s.output_shapes = s.output_shapes_including_background.clone();
    Ok(())
}

pub fn use_multicolor_shapes(s: &mut SolverState) -> Res<()> {
    s.shapes = s.multicolor_shapes.clone();
    Ok(())
}

pub fn shapes_to_number_sequence(s: &mut SolverState, i: usize) -> Res<()> {
    if i == 0 {
        s.number_sequences = vec![vec![]; s.images.len()];
        s.output_number_sequences = vec![vec![]; s.output_images.len()];
    }
    // The input is easy. Just number the shapes.
    s.number_sequences[i] = (0..s.shapes[i].len() as i32).collect();
    if i < s.output_shapes.len() {
        // The output is harder. We need to find the shapes in the output image.
        // Then identify which of the input shapes they correspond to.
        s.output_number_sequences[i] = s.output_shapes[i]
            .iter()
            .map(|output_shape| {
                output_shape
                    .find_matching_shape_index(&s.shapes[i])
                    .map(|index| index as i32)
                    .unwrap_or(-1)
            })
            .collect();
    }
    Ok(())
}

pub fn solve_number_sequence(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(s.number_sequences);
    must_not_be_empty!(s.output_number_sequences);
    // Use the longest sequence as the model.
    let longest_sequence = s
        .output_number_sequences
        .iter()
        .enumerate()
        .max_by_key(|(_i, seq)| seq.len())
        .map(|(i, _seq)| i)
        .ok_or(err!("no sequences"))?;
    let model = &s.output_number_sequences[longest_sequence];
    // Check that the model matches all sequences.
    for seq in &s.output_number_sequences {
        for j in 0..seq.len() {
            if seq[j] != model[j] {
                return Err(err!("sequences don't match"));
            }
        }
    }
    // Apply the model to all sequences.
    for seq in &mut s.number_sequences {
        *seq = model.clone();
    }
    Ok(())
}

/// Put the shapes after each other from left to right.
pub fn number_sequence_to_shapes_left_to_right(s: &mut SolverState, i: usize) -> Res<()> {
    must_not_be_empty!(s.number_sequences);
    let mut shapes: Shapes = vec![];
    let mut next_x = 0;
    for &index in &s.number_sequences[i] {
        if index < 0 {
            return Err(err!("invalid index"));
        }
        let shape = s.shapes[i]
            .get(index as usize)
            .ok_or(err!("no shape for index"))?
            .move_to(Vec2 { x: next_x, y: 0 });
        next_x += shape.bb.width();
        shapes.push(shape.into());
    }
    s.shapes[i] = shapes;
    Ok(())
}

pub fn follow_shape_sequence_per_output(s: &mut SolverState) -> Res<()> {
    // I've wrapped these in one step because there is nothing else to do with number sequences at the moment.
    s.apply(shapes_to_number_sequence)?;
    solve_number_sequence(s)?;
    s.apply(number_sequence_to_shapes_left_to_right)
}

pub fn remap_colors_per_output(s: &mut SolverState) -> Res<()> {
    let mut color_map: Vec<i32> = vec![-1; COLORS.len()];
    for i in 0..s.output_images.len() {
        let input = &s.images[i];
        let output = &s.output_images[i];
        let (w, h) = tools::width_and_height(input);
        let (ow, oh) = tools::width_and_height(output);
        if w != ow || h != oh {
            return Err(err!("images have different sizes"));
        }
        for y in 0..h {
            for x in 0..w {
                let input_color = input[(x as usize, y as usize)];
                let output_color = output[(x as usize, y as usize)];
                if color_map[input_color as usize] == -1 {
                    color_map[input_color as usize] = output_color;
                } else if color_map[input_color as usize] != output_color {
                    return Err(err!("no consistent mapping"));
                }
            }
        }
    }
    for image in &mut s.images {
        let mut new_image = (**image).clone();
        new_image.update(|_x, _y, cell| {
            let c = color_map[cell as usize];
            if c == -1 {
                cell
            } else {
                c
            }
        });
        *image = new_image.into();
    }
    Ok(())
}

/// Maps everything to s.colors[0]. Restores the original colors at the end.
/// For example, if colors[0] is [red, blue], and colors[1] is [purple, green],
/// then in image 1 purple will be mapped to red, and green will be mapped to blue.
/// Also affects the output images.
pub fn use_relative_colors(s: &mut SolverState) -> Res<()> {
    let mut new_images: ImagePerExample = vec![s.images[0].clone()];
    let mut any_modified = false;
    for i in 1..s.images.len() {
        if s.colors[i] == s.colors[0] {
            new_images.push(s.images[i].clone());
        } else {
            any_modified = true;
            new_images
                .push(tools::map_colors_in_image(&s.images[i], &s.colors[i], &s.colors[0]).into());
        }
    }
    if !any_modified {
        return Err(err!("colors already uniform"));
    }
    let mut new_output_images: ImagePerExample = vec![];
    for i in 0..s.output_images.len() {
        new_output_images.push(
            tools::map_colors_in_image(&s.output_images[i], &s.colors[i], &s.colors[0]).into(),
        );
    }
    let original_colors = s.colors.clone();
    s.colors = vec![s.colors[0].clone(); s.colors.len()];
    s.output_images = new_output_images;
    s.init_from_images(new_images);
    s.add_finishing_step(move |s: &mut SolverState| {
        let mut new_images: ImagePerExample = vec![s.images[0].clone()];
        for i in 1..s.images.len() {
            new_images.push(
                tools::map_colors_in_image(&s.images[i], &original_colors[0], &original_colors[i])
                    .into(),
            );
        }
        s.images = new_images;
        Ok(())
    });
    Ok(())
}

pub fn discard_small_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let mut new_shapes = vec![];
    for shape in &s.shapes[i] {
        if shape.bb.width() > 1 && shape.bb.height() > 1 {
            new_shapes.push(shape.clone());
        }
    }
    s.shapes[i] = new_shapes;
    Ok(())
}

pub fn erase_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let mut image = (*s.images[i]).clone();
    for shape in &s.shapes[i] {
        image.erase_shape(shape);
    }
    s.images[i] = image.into();
    Ok(())
}

/// Finds the best placement for each shape so it lines up with the most most pixels in the image.
pub fn place_shapes_best_match_with_all_transforms(s: &mut SolverState, i: usize) -> Res<()> {
    // TODO: This is kinda slow. Find a way to error out when it's useless.
    let image = &s.images[i];
    let mut new_shapes = vec![];
    for shape in &s.shapes[i] {
        let mut variations = vec![shape.clone()];
        for _ in 0..3 {
            variations.push(variations.last().unwrap().rotate_90_cw().into());
        }
        variations.push(variations.last().unwrap().flip_horizontal().into());
        for _ in 0..4 {
            variations.push(variations.last().unwrap().rotate_90_cw().into());
        }
        let (i, place) = variations
            .iter()
            .map(|shape| tools::place_shape(image, shape))
            .enumerate()
            .rev()
            .max_by_key(|(_i, place)| place.as_ref().map(|p| p.match_count).unwrap_or(0))
            .unwrap();
        let place = place?;
        new_shapes.push(
            variations[i]
                .move_by(place.pos - variations[i].bb.top_left())
                .into(),
        );
    }
    s.shapes[i] = new_shapes;
    Ok(())
}

pub fn place_shapes_best_match_with_just_translation(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &s.images[i];
    let mut new_shapes = vec![];
    for shape in &s.shapes[i] {
        let place = tools::place_shape(image, shape)?;
        new_shapes.push(shape.move_by(place.pos - shape.bb.top_left()).into());
    }
    s.shapes[i] = new_shapes;
    Ok(())
}

pub fn keep_only_border_lines(s: &mut SolverState, i: usize) -> Res<()> {
    let mut new_lines = tools::LineSet {
        horizontal: vec![],
        vertical: vec![],
    };
    let mut any_removed = false;
    let (width, height) = s.width_and_height(i);
    for line in &s.lines[i].horizontal {
        if line.pos == 0 || line.pos + line.width as i32 == height {
            new_lines.horizontal.push(*line);
        } else {
            any_removed = true;
        }
    }
    for line in &s.lines[i].vertical {
        if line.pos == 0 || line.pos + line.width as i32 == width {
            new_lines.vertical.push(*line);
        } else {
            any_removed = true;
        }
    }
    if !any_removed {
        return Err("no change");
    }
    s.lines[i] = new_lines.into();
    Ok(())
}

pub fn make_image_symmetrical(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &s.images[i];
    let mut new_image = (**image).clone();
    new_image.try_update(|x, y, mut c| {
        c = tools::blend_if_same_color(c, image[((image.width - 1 - x) as usize, y as usize)])?;
        c = tools::blend_if_same_color(c, image[(x as usize, (image.height - 1 - y) as usize)])?;
        tools::blend_if_same_color(
            c,
            image[(
                (image.width - 1 - x) as usize,
                (image.height - 1 - y) as usize,
            )],
        )
    })?;
    s.images[i] = new_image.into();
    Ok(())
}

pub fn make_image_rotationally_symmetrical(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &s.images[i];
    if image.width != image.height {
        return Err(err!("image not square"));
    }
    let mut new_image = (**image).clone();
    new_image.try_update(|x, y, mut c| {
        c = tools::blend_if_same_color(c, image[((image.height - 1 - y) as usize, x as usize)])?;
        c = tools::blend_if_same_color(
            c,
            image[(
                (image.width - 1 - x) as usize,
                (image.height - 1 - y) as usize,
            )],
        )?;
        tools::blend_if_same_color(c, image[(y as usize, (image.width - 1 - x) as usize)])
    })?;
    s.images[i] = new_image.into();
    Ok(())
}

/// Removes columns that match the previous column.
pub fn deduplicate_horizontally(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &s.images[i];
    let (width, height) = s.width_and_height(i);
    let mut is_duplicate = vec![true; width as usize];
    is_duplicate[0] = false;
    for x in 1..width {
        for y in 0..height {
            if image[(x as usize, y as usize)] != image[((x - 1) as usize, y as usize)] {
                is_duplicate[x as usize] = false;
                break;
            }
        }
    }
    if !is_duplicate.iter().any(|&b| b) {
        return Err("no change");
    }
    let mut new_image = vec![];
    for y in 0..height {
        let mut row = vec![];
        for x in 0..width {
            if !is_duplicate[x as usize] {
                row.push(image[(x as usize, y as usize)]);
            }
        }
        new_image.push(row);
    }
    s.images[i] = Image::from_vecvec(new_image).into();
    Ok(())
}

/// Removes rows that match the previous row.
pub fn deduplicate_vertically(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &s.images[i];
    let (width, height) = s.width_and_height(i);
    let mut is_duplicate = vec![true; height as usize];
    is_duplicate[0] = false;
    for y in 1..height {
        for x in 0..width {
            if image[(x as usize, y as usize)] != image[(x as usize, (y - 1) as usize)] {
                is_duplicate[y as usize] = false;
                break;
            }
        }
    }
    if !is_duplicate.iter().any(|&b| b) {
        return Err("no change");
    }
    let mut new_image = vec![];
    for y in 0..height {
        if !is_duplicate[y as usize] {
            let mut row = vec![];
            for x in 0..width {
                row.push(image[(x as usize, y as usize)]);
            }
            new_image.push(row);
        }
    }
    s.images[i] = Image::from_vecvec(new_image).into();
    Ok(())
}

pub fn make_common_output_image(s: &mut SolverState) -> Res<()> {
    s.output_width_and_height_all()?;
    let mut new_output_image = (*s.output_images[0]).clone();
    new_output_image.update(|x, y, mut c| {
        for i in 1..s.output_images.len() {
            if c != s.output_images[i][(x, y)] {
                c = 0;
                break;
            }
        }
        c
    });
    let new_output_image: Rc<Image> = new_output_image.into();
    s.images = s.images.iter().map(|_| new_output_image.clone()).collect();
    Ok(())
}

/// Takes the output shapes from all outputs. Deduplicates them.
pub fn take_all_shapes_from_output(s: &mut SolverState) -> Res<()> {
    let mut new_shapes: Shapes = vec![];
    for shapes in &s.output_shapes {
        for shape in shapes {
            if !new_shapes.iter().any(|s| s.pixels == shape.pixels) {
                new_shapes.push(shape.clone());
            }
        }
    }
    s.shapes = vec![new_shapes; s.images.len()];
    Ok(())
}

struct CoverageState {
    image: Rc<Image>,
    shapes: Shapes,
    is_covered: Vec<Vec<bool>>,
    still_uncovered: usize,
    // (shape index, position)
    placements: Vec<(usize, Vec2)>,
    budget: i32,
}

pub fn cover_image_with_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &s.images[i];
    let shapes = &s.shapes[i];
    // We assume it's not a terribly hard problem. But we still allow backtracking.
    let mut state = CoverageState {
        image: image.clone(),
        shapes: shapes.clone(),
        is_covered: vec![vec![false; image.width]; image.height],
        still_uncovered: image.colors_iter().filter(|&c| c != 0).count(),
        placements: vec![],
        budget: 1000,
    };
    cover_image_with_shapes_recursive(&mut state, 0)?;
    let mut new_shapes = vec![];
    for (i, pos) in state.placements {
        let shape = state.shapes[i].move_to(pos);
        new_shapes.push(shape.into());
    }
    s.shapes[i] = new_shapes;
    Ok(())
}

/// Places one shape in a possible position, then recurses.
fn cover_image_with_shapes_recursive(mut state: &mut CoverageState, min_y: i32) -> Res<()> {
    if state.still_uncovered == 0 {
        return Ok(());
    }
    let (width, height) = tools::width_and_height(&state.image);
    for y in min_y..height {
        for x in 0..width {
            'next: for i in 0..state.shapes.len() {
                let shape = state.shapes[i].clone();
                if shape.color() == 0 {
                    continue;
                }
                // Check if it can be placed here.
                for cell in &*shape.pixels {
                    if state.image.get_or(cell.x + x, cell.y + y, 0) == 0 {
                        continue 'next;
                    }
                    if state.is_covered[(cell.y + y) as usize][(cell.x + x) as usize] {
                        continue 'next;
                    }
                }
                // Place it here.
                state.placements.push((i, Vec2 { x, y }));
                for cell in &*shape.pixels {
                    state.is_covered[(cell.y + y) as usize][(cell.x + x) as usize] = true;
                    state.still_uncovered -= 1;
                }
                if let Ok(()) = cover_image_with_shapes_recursive(&mut state, y) {
                    return Ok(());
                }
                // It didn't pan out. Undo the placement if we still have budget.
                if state.budget <= 0 {
                    return Err(err!("budget exhausted"));
                }
                state.placements.pop();
                for cell in &*shape.pixels {
                    state.is_covered[(cell.y + y) as usize][(cell.x + x) as usize] = false;
                    state.still_uncovered += 1;
                }
            }
        }
    }
    state.budget -= 1;
    Err(err!("could not find covering"))
}

pub fn order_colors_and_shapes_by_output_frequency_increasing(s: &mut SolverState) -> Res<()> {
    let mut color_counts = vec![0; COLORS.len()];
    for image in &s.output_images {
        tools::count_colors_in_image(image, &mut color_counts);
    }
    let mut color_order: Vec<i32> = (0..COLORS.len() as i32).collect();
    color_order.sort_by_key(|&c| color_counts[c as usize]);
    for i in 0..s.images.len() {
        s.colors[i] = color_order.clone();
        let mut new_shapes = s.shapes[i].clone();
        new_shapes.sort_by_key(|shape| color_counts[shape.color() as usize]);
        s.shapes[i] = new_shapes;
    }
    Ok(())
}

pub fn atomize_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let mut new_shapes = vec![];
    for shape in &s.shapes[i] {
        for cell in shape.cells() {
            new_shapes.push(Shape::new(vec![cell]).into());
        }
    }
    s.shapes[i] = new_shapes;
    Ok(())
}

/// Expands each pixel in the shapes to a vertical or horizontal infinite line of the same color.
/// The direction and order of the colors is gleaned from the output image.
/// The lines are drawn on the image.
pub fn dots_to_lines_per_output(s: &mut SolverState) -> Res<()> {
    let colors = tools::get_used_colors(&s.images);
    if colors.len() > 3 {
        return Err(err!("too many colors"));
    }
    // Figure out which is horizontal and which is vertical.
    let mut horizontal_count = vec![0; COLORS.len()];
    let mut vertical_count = vec![0; COLORS.len()];
    for i in 0..s.output_images.len() {
        let image = &s.output_images[i];
        let (w, h) = tools::width_and_height(image);
        for shape in &s.shapes[i] {
            for tools::Pixel { x, y, color } in shape.cells() {
                if x < 0 || x >= w || y < 0 || y >= h {
                    return Err(err!("shape out of bounds"));
                }
                for ix in 0..w {
                    if image[(ix as usize, y as usize)] == color {
                        horizontal_count[color as usize] += 1;
                    }
                }
                for iy in 0..h {
                    if image[(x as usize, iy as usize)] == color {
                        vertical_count[color as usize] += 1;
                    }
                }
            }
        }
    }
    let is_horizontal = horizontal_count
        .iter()
        .zip(&vertical_count)
        .map(|(h, v)| h > v)
        .collect();
    // Figure out the order of colors.
    'order: for order in tools::possible_orders(&colors) {
        let reverse_colors = tools::reverse_colors(&order);
        for i in 0..s.output_images.len() {
            let image = &s.output_images[i];
            let mut shapes = s.shapes[i].clone();
            shapes.sort_by_key(|s| reverse_colors[s.color() as usize]);
            let (w, h) = tools::width_and_height(image);
            let mut new_image = Image::new(w as usize, h as usize);
            dots_to_lines(&shapes, &is_horizontal, &mut new_image);
            if !tools::a_matches_b_where_a_is_not_transparent(&new_image, image) {
                continue 'order;
            }
        }
        // This order matches all outputs. Draw it.
        for i in 0..s.images.len() {
            let mut new_image = (*s.images[i]).clone();
            let mut shapes = s.shapes[i].clone();
            shapes.sort_by_key(|s| reverse_colors[s.color() as usize]);
            dots_to_lines(&shapes, &is_horizontal, &mut new_image);
            s.images[i] = new_image.into();
        }
        return Ok(());
    }
    Err(err!("could not figure out dots"))
}

pub fn dots_to_lines(shapes: &Shapes, is_horizontal: &Vec<bool>, image: &mut Image) {
    let (w, h) = tools::width_and_height(image);
    for shape in shapes {
        for tools::Pixel { x, y, color } in shape.cells() {
            if is_horizontal[color as usize] {
                for ix in 0..w {
                    image[(ix as usize, y as usize)] = color;
                }
            } else {
                for iy in 0..h {
                    image[(x as usize, iy as usize)] = color;
                }
            }
        }
    }
}

/// Deletes a few pixels from colors that are otherwise unused.
pub fn delete_noise(s: &mut SolverState, i: usize) -> Res<()> {
    let cs = &s.colorsets[i];
    let counts: Vec<usize> = cs.iter().map(|c| c.pixels.len()).collect();
    let total: usize = counts.iter().sum();
    let small: Vec<&Rc<Shape>> = cs
        .iter()
        .filter(|shape| (**shape).pixels.len() <= total / 8)
        .collect();
    if small.is_empty() {
        return Err("no change");
    }
    let mut new_image = (*s.images[i]).clone();
    for shape in small {
        new_image.erase_shape(shape);
    }
    s.images[i] = new_image.into();
    Ok(())
}

/// Makes the image a square by expanding the canvas up and to the left.
pub fn make_square_up_left(s: &mut SolverState, i: usize) -> Res<()> {
    let (w, h) = s.width_and_height(i);
    if h < w {
        let mut new_image = Image::new(w as usize, w as usize);
        new_image.draw_image_at(&s.images[i], Vec2 { x: 0, y: w - h });
        s.images[i] = new_image.into();
        return Ok(());
    } else if w < h {
        let mut new_image = Image::new(h as usize, h as usize);
        new_image.draw_image_at(&s.images[i], Vec2 { x: h - w, y: 0 });
        s.images[i] = new_image.into();
        return Ok(());
    }
    Err("no change")
}

pub fn shrink_to_output_size_from_top_left(s: &mut SolverState, i: usize) -> Res<()> {
    let (ow, oh) = s.output_width_and_height_all()?;
    let (w, h) = s.width_and_height(i);
    if w < ow || h < oh {
        println!("{} < {} or {} < {}", w, ow, h, oh);
        return Err("Output is larger");
    } else if w == ow && h == oh {
        return Err("no change");
    }
    let mut new_image = Image::new(ow as usize, oh as usize);
    new_image.draw_image_at(
        &s.images[i],
        Vec2 {
            x: ow - w,
            y: oh - h,
        },
    );
    s.images[i] = new_image.into();
    Ok(())
}

pub fn recolor_shapes_to_selected_color(s: &mut SolverState, i: usize) -> Res<()> {
    let new_color = s.colors[i][0];
    let mut any_change = false;
    for shapes in &mut s.shapes {
        for shape in shapes {
            if shape.color() != new_color {
                any_change = true;
            }
            *shape = shape.recolor(new_color).into();
        }
    }
    if !any_change {
        return Err("no change");
    }
    Ok(())
}
pub fn zoom_to_content(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &s.images[i];
    // Work from the bounding boxes of colorsets.
    let mut bb = tools::Rect::empty();
    for cs in &s.colorsets[i] {
        bb.top = bb.top.min(cs.bb.top);
        bb.left = bb.left.min(cs.bb.left);
        bb.bottom = bb.bottom.max(cs.bb.bottom);
        bb.right = bb.right.max(cs.bb.right);
    }
    if bb.top == 0
        && bb.left == 0
        && bb.bottom == image.height as i32
        && bb.right == image.width as i32
    {
        return Err("no change");
    }
    if bb.bottom < 0 {
        return Err(err!("no content"));
    }
    let new_image = image.subimage(
        bb.left as usize,
        bb.top as usize,
        bb.width() as usize,
        bb.height() as usize,
    );
    s.images[i] = new_image.into();
    s.shift_shapes(
        i,
        Vec2 {
            x: -bb.left,
            y: -bb.top,
        },
    );
    Ok(())
}
pub fn extend_zoom_up_left_until_square(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &s.images[i];
    if image.width < image.height {
        let extension = image.height - image.width;
        if image.left < extension {
            return Err("no change");
        }
        let mut new_image = (**image).clone();
        new_image.width += extension;
        new_image.left -= extension;
        s.images[i] = new_image.into();
        s.shift_shapes(
            i,
            Vec2 {
                x: extension as i32,
                y: 0,
            },
        );
        Ok(())
    } else if image.width > image.height {
        let extension = image.width - image.height;
        if image.top < extension {
            return Err("no change");
        }
        let mut new_image = (**image).clone();
        new_image.height += extension;
        new_image.top -= extension;
        s.images[i] = new_image.into();
        s.shift_shapes(
            i,
            Vec2 {
                x: 0,
                y: extension as i32,
            },
        );
        Ok(())
    } else {
        Err("no change")
    }
}
pub fn recolor_image_to_selected_color(s: &mut SolverState, i: usize) -> Res<()> {
    let new_color = s.colors[i][0];
    let mut any_change = false;
    let mut new_image = (*s.images[i]).clone();
    for y in 0..new_image.height {
        for x in 0..new_image.width {
            let c = new_image[(x, y)];
            if c != 0 && c != new_color {
                any_change = true;
                new_image[(x, y)] = new_color;
            }
        }
    }
    if !any_change {
        return Err("no change");
    }
    s.images[i] = new_image.into();
    Ok(())
}
pub fn reset_zoom(s: &mut SolverState, i: usize) -> Res<()> {
    if !s.images[i].is_zoomed() {
        return Err("no change");
    }
    let mut new_image = (*s.images[i]).clone();
    s.shift_shapes(
        i,
        Vec2 {
            x: new_image.left as i32,
            y: new_image.top as i32,
        },
    );
    new_image.top = 0;
    new_image.left = 0;
    new_image.width = new_image.full_width;
    new_image.height = new_image.full_height;
    s.images[i] = new_image.into();
    Ok(())
}

pub fn crop_to_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &mut s.images[i];
    let shape = s.shapes[i].first().ok_or(err!("no shape"))?;
    if shape.bb.left <= 0
        && shape.bb.top <= 0
        && shape.bb.right >= image.width as i32
        && shape.bb.bottom >= image.height as i32
    {
        return Err("no change");
    }
    *image = image
        .crop(
            shape.bb.left,
            shape.bb.top,
            shape.bb.width(),
            shape.bb.height(),
        )
        .into();
    s.shift_shapes(i, -1 * shape.bb.top_left());
    Ok(())
}
pub fn inset_by_one(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &mut s.images[i];
    if image.width < 3 || image.height < 3 {
        return Err(err!("image too small"));
    }
    *image = image
        .crop(1, 1, image.width as i32 - 2, image.height as i32 - 2)
        .into();
    s.shift_shapes(i, Vec2 { x: -1, y: -1 });
    Ok(())
}

pub fn align_shapes_to_saved_shape_horizontal(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
    must_not_be_empty!(shapes);
    let saved_shapes: &ShapesPerExample = s.saved_shapes.last().ok_or(err!("no saved shapes"))?;
    let saved_shape: &Shape = saved_shapes[i].first().ok_or(err!("no saved shape"))?;
    let mut any_change = false;
    for shape in shapes {
        let size_diff = shape.bb.height() - saved_shape.bb.height();
        if size_diff % 2 != 0 {
            return Err(err!("size difference not even"));
        }
        let destination_y = saved_shape.bb.top - size_diff / 2;
        if shape.bb.top != destination_y {
            any_change = true;
            *shape = shape
                .move_by(Vec2 {
                    x: 0,
                    y: destination_y - shape.bb.top,
                })
                .into();
        }
    }
    if !any_change {
        return Err("no change");
    }
    Ok(())
}

pub fn drop_all_pixels_down(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &mut s.images[i];
    let mut new_image: Image = (**image).clone();
    let mut columns = vec![0; image.width];
    new_image.clear();
    for y in (0..image.height).rev() {
        for x in 0..image.width {
            let c = image[(x, y)];
            if c != 0 {
                new_image[(x, image.height - columns[x] - 1)] = c;
                columns[x] += 1;
            }
        }
    }
    *image = new_image.into();
    Ok(())
}
