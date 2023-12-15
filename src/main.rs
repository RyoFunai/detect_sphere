use futures::executor::LocalPool;
use futures::stream::StreamExt;
use futures::task::LocalSpawnExt;
use nalgebra::{DMatrix, DVector};
use r2r::sensor_msgs::msg::PointCloud2;
use r2r::std_msgs::msg::Header;
use r2r::visualization_msgs::msg::Marker;
use r2r::{Context, Node, QosProfile};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::time::Duration;

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
    z: f64,
}
#[derive(Debug, Clone, Copy)]

struct Sphere {
    center: Point,
    radius: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("START!!");
    let ctx: Context = Context::create()?;
    let mut node: Node = Node::create(ctx, "node", "namespace")?;

    let mut subscription =
        node.subscribe::<PointCloud2>("/hokuyo_cloud2", QosProfile::default())?;
    let cloud_publisher =
        node.create_publisher::<PointCloud2>("/my_cloud2", QosProfile::default())?;
    let sphere_publisher = node.create_publisher::<Marker>("/sphere", QosProfile::default())?;
    // ローカルプールエグゼキュータを作成(非同期)
    let mut pool: LocalPool = LocalPool::new();
    let spawner: futures::executor::LocalSpawner = pool.spawner();

    spawner.spawn_local(async move {
        while let Some(msg_arc) = subscription.next().await {
            // ArcからPointCloud2への参照を直接取得
            let msg: &PointCloud2 = &msg_arc;
            let mut src_points: Vec<Point> = convert_pc2_to_vec(msg);
            reduce_points(&mut src_points, 2.0, 1.5, 1.0);
            remove_plane(&mut src_points); //壁2つと床1つの平面を除去した。
            remove_plane(&mut src_points);
            remove_plane(&mut src_points);
            if src_points.len() > 4 {
                if let Some(sphere) = ransac_for_spheres(&mut src_points, 30, 0.03) {
                    // println!("{:#?}", sphere);
                    let sphere_marker = create_sphere_marker(&sphere);
                    sphere_publisher.publish(&sphere_marker).unwrap();
                };
            }
            let pub_msg: PointCloud2 = convert_vec_to_pc2(&mut src_points);
            cloud_publisher.publish(&pub_msg).unwrap();
        }
    })?;
    loop {
        node.spin_once(std::time::Duration::from_millis(100));
        pool.run_until_stalled();
    }
}

// ransacによる球体検出
fn ransac_for_spheres(
    points: &mut Vec<Point>,
    num_iterations: usize,
    fit_threshold: f64,
) -> Option<Sphere> {
    let mut best_sphere = None;
    let mut max_inliers = 0;

    for _ in 0..num_iterations {
        if let Some(sample_points) = get_random_points_within_diameter(&points, 4, 0.25) {
            if let Some(sphere) = fit_sphere_with_radius(&sample_points) {
                let inliers_count = points
                    .iter()
                    .filter(|&p| is_point_near_sphere(p, &sphere, fit_threshold))
                    .count();

                if inliers_count > max_inliers {
                    max_inliers = inliers_count;
                    best_sphere = Some(sphere);
                }
            } else {
                println!("can't detect sphere!!");
            }
        }
    }
    best_sphere
}


// 既定の半径からフィッティング
fn fit_sphere_with_radius(points: &Vec<Point>) -> Option<Sphere> {
    if points.len() != 4 {
        return None;
    }

    let radius: f64 = 0.19 / 2.0; // ボールのサイズ
    let mut a = DMatrix::zeros(4, 4);
    let mut b = DVector::zeros(4);

    for (i, point) in points.iter().enumerate() {
        a[(i, 0)] = -2.0 * point.x;
        a[(i, 1)] = -2.0 * point.y;
        a[(i, 2)] = -2.0 * point.z;
        a[(i, 3)] = 1.0;
        b[i] = radius.powi(2) - point.x.powi(2) - point.y.powi(2) - point.z.powi(2);
    }

    // 線形システムを解く
    if let Some(sol) = a.lu().solve(&b) {
        let center = Point {
            x: sol[0],
            y: sol[1],
            z: sol[2],
        };
        Some(Sphere { center, radius })
    } else {
        None
    }
}

// inlierかどうかを判断
fn is_point_near_sphere(point: &Point, sphere: &Sphere, threshold: f64) -> bool {
    let distance = ((point.x - sphere.center.x).powi(2)
        + (point.y - sphere.center.y).powi(2)
        + (point.z - sphere.center.z).powi(2))
    .sqrt();
    (distance - sphere.radius).abs() <= threshold
}

// ransacによる平面検出&削除
fn remove_plane(points: &mut Vec<Point>) -> Vec<Point> {
    let max_iterations: i32 = 50;
    let mut max_inliers_count = 0;
    let mut best_a: f64 = 0.0;
    let mut best_b: f64 = 0.0;
    let mut best_c: f64 = 0.0;
    let mut best_d: f64 = 0.0;
    let distance_threshold: f64 = 0.05;

    for _ in 0..max_iterations {
        if let Some(random_points) = get_random_points(&points, 3) {
            let p1 = random_points[0];
            let p2 = random_points[1];
            let p3 = random_points[2];
            let mut a: f64 = 0.0;
            let mut b: f64 = 0.0;
            let mut c: f64 = 0.0;
            let mut d: f64 = 0.0;
            compute_plane_coeff(&p1, &p2, &p3, &mut a, &mut b, &mut c, &mut d);

            let inliers_count = points
                .iter()
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

    points.retain(|pt| {
        distance_to_plane(&pt, &best_a, &best_b, &best_c, &best_d) > distance_threshold
    });

    points.to_vec()
}

// 平面の式の係数を算出
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

// 平面までの距離
fn distance_to_plane(pt: &Point, a: &f64, b: &f64, c: &f64, d: &f64) -> f64 {
    return (a * pt.x + b * pt.y + c * pt.z + d).abs() / (a * a + b * b + c * c).sqrt();
}

// 範囲外の点群を消去する。
fn reduce_points(points: &mut Vec<Point>, x_limit: f64, y_limit: f64, z_limit: f64) -> &Vec<Point> {
    points.retain(|point| {
        point.x.abs() <= x_limit && point.y.abs() <= y_limit && point.z.abs() <= z_limit
    });
    points
}

// num_pointsの数、ランダムにサンプリング
fn get_random_points(points: &Vec<Point>, num_points: usize) -> Option<Vec<Point>> {
    let mut rng = thread_rng();
    if points.len() < num_points {
        return None;
    }

    let sample_points: Vec<Point> = points
        .choose_multiple(&mut rng, num_points)
        .cloned()
        .collect();

    if sample_points.len() == num_points {
        Some(sample_points)
    } else {
        None
    }
}

// 球体でないところの点群を取得すると無駄に処理してしまう。その対策として、点群が一定距離より遠い場合はリサンプリング。
fn get_random_points_within_diameter(
    points: &Vec<Point>,
    num_points: usize,
    max_diameter: f64,
) -> Option<Vec<Point>> {
    let mut rng = thread_rng();

    for _ in 0..100 {
        // ループの回数は適宜調整
        if points.len() < num_points {
            return None;
        }

        let sample_points: Vec<Point> = points
            .choose_multiple(&mut rng, num_points)
            .cloned()
            .collect();

        if sample_points.len() == num_points && is_within_diameter(&sample_points, max_diameter) {
            return Some(sample_points);
        }
    }

    None
}

fn is_within_diameter(points: &Vec<Point>, max_diameter: f64) -> bool {
    for (i, point1) in points.iter().enumerate() {
        for point2 in points.iter().skip(i + 1) {
            if get_distance(point1, point2) > max_diameter {
                return false;
            }
        }
    }
    true
}

fn get_distance(p1: &Point, p2: &Point) -> f64 {
    ((p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2) + (p1.z - p2.z).powi(2)).sqrt()
}

// pointcloud2からベクターへ変換
fn convert_pc2_to_vec(cloud: &PointCloud2) -> Vec<Point> {
    // 点の数
    let num_points: usize = (cloud.width * cloud.height) as usize;
    // 保持する点データを作成
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

fn convert_vec_to_pc2(points: &mut Vec<Point>) -> PointCloud2 {
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

//rviz2で球体を表示するためのマーカーを作成
fn create_sphere_marker(sphere: &Sphere) -> Marker {
    let mut marker = Marker::default();
    marker.header.frame_id = "map".to_string();
    let mut clock = r2r::Clock::create(r2r::ClockType::RosTime).unwrap();
    let now: Duration = clock.get_now().unwrap(); // 現在のROS時刻を取得

    marker.header.stamp.sec = now.as_secs() as i32;
    marker.header.stamp.nanosec = now.subsec_nanos();

    marker.ns = "spheres".to_string();
    marker.id = 0;
    marker.type_ = 2; //2番が球体
    marker.action = 0;

    marker.pose.position.x = sphere.center.x;
    marker.pose.position.y = sphere.center.y;
    marker.pose.position.z = sphere.center.z;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = sphere.radius * 2.0; // 球体の直径
    marker.scale.y = sphere.radius * 2.0;
    marker.scale.z = sphere.radius * 2.0;

    marker.color.a = 1.0; // 不透明度
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker
}
