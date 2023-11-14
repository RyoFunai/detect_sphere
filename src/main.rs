use futures::executor::LocalPool;
use futures::future;
use futures::stream::StreamExt;
use futures::task::LocalSpawnExt;
use r2r::sensor_msgs::msg::PointCloud2;
use r2r::std_msgs::msg::Header;
use r2r::{Context, Node, Publisher, QosProfile};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::sync::Arc;
use std::time::Duration;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("START!!");
    let ctx: Context = Context::create()?;
    let mut node: Node = Node::create(ctx, "node", "namespace")?;

    let mut subscription =
        node.subscribe::<PointCloud2>("/hokuyo_cloud2", QosProfile::default())?;
    let publisher = node.create_publisher::<PointCloud2>("/my_cloud2", QosProfile::default())?;

    // ローカルプールエグゼキュータを作成(非同期)
    let mut pool: LocalPool = LocalPool::new();
    let spawner: futures::executor::LocalSpawner = pool.spawner();

    spawner.spawn_local(async move {
        while let Some(msg_arc) = subscription.next().await {
            // ArcからPointCloud2への参照を直接取得
            let msg: &PointCloud2 = &msg_arc;
            let mut src_points = convert_pointcloud2_to_points(msg);
            println!("PointCloud2 size> {}", src_points.len());
            reduce_points(&mut src_points, 2.0, 1.5, 1.0);
            println!("reduced size    > {}", src_points.len());
            remove_plane_reduce(&mut src_points);
            remove_plane_reduce(&mut src_points);
            remove_plane_reduce(&mut src_points);
            let pub_msg: PointCloud2 = vec_to_pc2(&mut src_points);
            publisher.publish(&pub_msg).unwrap();
        }
    })?;

    loop {
        node.spin_once(std::time::Duration::from_millis(100));
        pool.run_until_stalled();
    }
}

fn remove_plane(points: &mut Vec<Point>) -> Vec<Point> {
    let max_iterations: i32 = 50;
    let mut max_iterations_count = 0;
    let mut best_a: f64 = 0.0;
    let mut best_b: f64 = 0.0;
    let mut best_c: f64 = 0.0;
    let mut best_d: f64 = 0.0;
    let distance_threshold: f64 = 0.04;

    for i in 0..max_iterations {
        let Some((p1, p2, p3)) = get_three_random_points((&points).to_vec()) else {
            todo!()
        };
        println!("Random Points: {:?}, {:?}, {:?}", p1, p2, p3);

        let mut a: f64 = 0.0;
        let mut b: f64 = 0.0;
        let mut c: f64 = 0.0;
        let mut d: f64 = 0.0;
        compute_plane_coeff(&p1, &p2, &p3, &mut a, &mut b, &mut c, &mut d);

        let mut inliers_count = 0;
        for pt in &mut *points {
            if distance_to_plane(&pt, &a, &b, &c, &d) <= distance_threshold {
                inliers_count += 1;
            }
        }
        if inliers_count > max_iterations_count {
            max_iterations_count = inliers_count;
            best_a = a;
            best_b = b;
            best_c = c;
            best_d = d;
        }
    }
    let mut filtered_points: Vec<Point> = vec![];
    for pt in &mut *points {
        if distance_to_plane(&pt, &best_a, &best_b, &best_c, &best_d) > distance_threshold {
            filtered_points.push(*pt);
        }
    }
    return filtered_points;
}

fn remove_plane_reduce(points: &mut Vec<Point>) -> Vec<Point> {
    let max_iterations: i32 = 50;
    let mut max_inliers_count = 0;
    let mut best_a: f64 = 0.0;
    let mut best_b: f64 = 0.0;
    let mut best_c: f64 = 0.0;
    let mut best_d: f64 = 0.0;
    let distance_threshold: f64 = 0.05;

    for _ in 0..max_iterations {
        if let Some((p1, p2, p3)) = get_three_random_points((&points).to_vec()) {
            let mut a: f64 = 0.0;
            let mut b: f64 = 0.0;
            let mut c: f64 = 0.0;
            let mut d: f64 = 0.0;
            compute_plane_coeff(&p1, &p2, &p3, &mut a, &mut b, &mut c, &mut d);

            let inliers_count = points.iter()
                .filter(|&pt| distance_to_plane(&pt, &a, &b, &c, &d) <= distance_threshold)
                .count();

            if inliers_count > max_inliers_count {
                max_inliers_count = inliers_count;
                best_a = a;
                best_b = b;
                best_c = c;
                best_d = d;
            }
        }
    }

    points.retain(|pt| distance_to_plane(&pt, &best_a, &best_b, &best_c, &best_d) > distance_threshold);

    points.to_vec()
}


fn compute_plane_coeff(
    p1: &Point,
    p2: &Point,
    p3: &Point,
    a: &mut f64,
    b: &mut f64,
    c: &mut f64,
    d: &mut f64,
) {
    *a = (p2.y - p1.y) * (p3.z - p1.z) - (p2.z - p1.z) * (p3.y - p1.y);
    *b = (p2.z - p1.z) * (p3.x - p1.x) - (p2.x - p1.x) * (p3.z - p1.z);
    *c = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
    *d = -1.0 * (*a * p1.x + *b * p1.y + *c * p1.z);
}

fn distance_to_plane(pt: &Point, a: &f64, b: &f64, c: &f64, d: &f64) -> f64 {
    return (a * pt.x + b * pt.y + c * pt.z + d).abs() / (a * a + b * b + c * c).sqrt();
}

fn reduce_points(points: &mut Vec<Point>, x_limit: f64, y_limit: f64, z_limit: f64) -> &Vec<Point> {
    points.retain(|point| {
        point.x.abs() <= x_limit && point.y.abs() <= y_limit && point.z.abs() <= z_limit
    });
    points
}

fn get_three_random_points(points: Vec<Point>) -> Option<(Point, Point, Point)> {
    let mut rng = thread_rng();
    let sample_points: Vec<&Point> = points.choose_multiple(&mut rng, 3).collect();

    if sample_points.len() == 3 {
        Some((*sample_points[0], *sample_points[1], *sample_points[2]))
    } else {
        None // ベクトルの長さが3未満の場合はNoneを返す
    }
}

fn convert_pointcloud2_to_points(cloud: &PointCloud2) -> Vec<Point> {
    // 点の数
    let num_points: usize = (cloud.width * cloud.height) as usize;
    // 点データを保持するためのベクトルを作成
    let mut points = Vec::with_capacity(num_points);

    // 各点に対するオフセットを計算（XYZがbit）
    let point_step = cloud.point_step as usize;
    let x_offset = cloud.fields[0].offset as usize;
    let y_offset = cloud.fields[1].offset as usize;
    let z_offset = cloud.fields[2].offset as usize;

    // 点を抽出
    for i in 0..num_points {
        let data_index = i * point_step;
        let x = f32::from_le_bytes(
            cloud.data[data_index + x_offset..data_index + x_offset + 4]
                .try_into()
                .unwrap(),
        ) as f64;
        let y = f32::from_le_bytes(
            cloud.data[data_index + y_offset..data_index + y_offset + 4]
                .try_into()
                .unwrap(),
        ) as f64;
        let z = f32::from_le_bytes(
            cloud.data[data_index + z_offset..data_index + z_offset + 4]
                .try_into()
                .unwrap(),
        ) as f64;
        points.push(Point { x, y, z });
    }

    points
}

fn vec_to_pc2(points: &mut Vec<Point>) -> PointCloud2 {
    let mut clock: r2r::Clock = r2r::Clock::create(r2r::ClockType::RosTime).unwrap();
    // 現在のROS時刻を取得
    let now = clock.get_now().unwrap();

    let mut cloud = PointCloud2 {
        header: Header {
            stamp: r2r::builtin_interfaces::msg::Time {
                sec: now.as_secs() as i32,
                nanosec: now.subsec_nanos(),
            },
            frame_id: "map".into(),
            ..Default::default()
        },
        height: 1,
        width: points.len() as u32,
        fields: vec![
            r2r::sensor_msgs::msg::PointField {
                name: "x".to_string(),
                offset: 0,
                datatype: 7, // FLOAT32
                count: 1,
            },
            r2r::sensor_msgs::msg::PointField {
                name: "y".to_string(),
                offset: 4,   // FLOAT32の次のバイトから
                datatype: 7, // FLOAT32
                count: 1,
            },
            r2r::sensor_msgs::msg::PointField {
                name: "z".to_string(),
                offset: 8,   // FLOAT32の2倍のバイトから
                datatype: 7, // FLOAT32
                count: 1,
            },
        ],

        is_bigendian: false,
        point_step: (std::mem::size_of::<f32>() * 3) as u32,
        row_step: ((std::mem::size_of::<f32>() * 3) * points.len()) as u32,
        data: vec![],
        is_dense: true,
        ..Default::default()
    };

    for point in points.iter() {
        // `.iter()` を使用してベクトルの要素にアクセス
        //Point(f64)→PC2(f32)
        cloud
            .data
            .extend_from_slice(&(point.x as f32).to_le_bytes());
        cloud
            .data
            .extend_from_slice(&(point.y as f32).to_le_bytes());
        cloud
            .data
            .extend_from_slice(&(point.z as f32).to_le_bytes());
    }

    cloud
}

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
    z: f64,
}
