    std::string scheme; 
    float sss;
    float bbb; 
    template<typename dist_t, class FeatVec_T>
    struct HNSWFinger {
        typedef FeatVec_T feat_vec_t;
        typedef Pair<dist_t, index_type> pair_t;
        typedef heap_t<pair_t, std::less<pair_t>> max_heap_t;
        typedef heap_t<pair_t, std::greater<pair_t>> min_heap_t;


        // scalar variables
        index_type num_node;
        index_type maxM;   // max number of out-degree for level l=1,...,L
        index_type maxM0;  // max number of out-degree for level l=0
        index_type efC;    // size of priority queue for construction time
        index_type max_level;
        index_type init_node;
        index_type subspace_dimension;  // dimension of each subspace in Product Quantization
        index_type sub_sample_points;   // number of sub-sampled points used to build quantizer subspace centors. 

        GraphL0<feat_vec_t> feature_vec;           // feature vectors only
        GraphL1 graph_l1;                       // neighborhood graphs from level 1 and above
        GraphFinger<dist_t, feat_vec_t> graph_l0_finger;   // Productquantized4Bits neighborhood graph built from graph_l0
        HNSWFinger() {
            std::string space_type = pecos::type_util::full_name<feat_vec_t>();
            //if (space_type != "pecos::ann::FeatVecDenseL2Simd<float>") {
            //    throw std::runtime_error("Currently, we only support L2 distance with float type.");
            //} 
        }
        ~HNSWFinger() {}
        struct Searcher : SetOfVistedNodes<unsigned short int> {
            typedef SetOfVistedNodes<unsigned short int> set_of_visited_nodes_t;
            typedef HNSWFinger<dist_t, FeatVec_T> hnswfinger_t;
            typedef heap_t<pair_t, std::less<pair_t>> max_heap_t;
            typedef heap_t<pair_t, std::greater<pair_t>> min_heap_t;

            const hnswfinger_t* hnsw;
            max_heap_t topk_queue;
            min_heap_t cand_queue;
            alignas(64) std::vector<float> query_projection;
            uint64_t query_rplsh_code;

            __m512i _lookup_table;// = _mm512_set1_epi64(talk2);
            alignas(64) std::vector<float> appx_dist;
            float query_norm;
            float query_squared_norm;
            //void (*approximate_distance)(size_t, const float&, const float&, const char*);
            bool which;
            Searcher(const hnswfinger_t* _hnsw=nullptr):
                SetOfVistedNodes<unsigned short int>(_hnsw? _hnsw->num_node : 0),
                hnsw(_hnsw)
            {}

            void reset() {
                set_of_visited_nodes_t::reset();
                topk_queue.clear();
                cand_queue.clear();
            }
            void setup_appx_results_containers() {
                query_projection.resize(hnsw->graph_l0_finger.finger.low_rank, 0);
                appx_dist.resize(hnsw->graph_l0_finger.max_degree % 16 == 0 ?  hnsw->graph_l0_finger.max_degree : (hnsw->graph_l0_finger.max_degree / 16 + 1) * 16, 0);
                //std::vector<float> cos_table{ 0.4788641, 0.44455987, 0.4081703, 0.3788809, 0.3529579, 0.3307132, 0.31037903, 0.29017207, 0.27027875, 0.25325617, 0.23492996, 0.21701016, 0.19880502, 0.1806895, 0.16216676, 0.14297655, 0.12419499, 0.104549415, 0.085694425, 0.066262625, 0.046003137, 0.02588392, 0.002810445, -0.01945299, -0.042384192, -0.066297606, -0.08810465, -0.11720696, -0.15735959, -0.19942293, -0.22520278};
                std::vector<float> cos_table(64);
                for (int i = 63; i >=0; i--) {
                    cos_table[i] = std::cos(i * ANGLE);
                    //if (cos_table[i - 16] >= 0 and cos_table[i - 16] < .245) {
                    //    cos_table[i - 16] *= .8;
                    //}
                    //else if (cos_table[i - 16] >= .245) {
                    //    cos_table[i - 16] *= 0.8;
                    //}
                    //else if (cos_table[i-16] < 0){
                    //    cos_table[i - 16] = 0;
                    //}
                }

                //std::vector<float> cos_table{ 0.6785726547241211, 0.6413908123970031, 0.6003756046295166, 0.5646432042121887, 0.5263587355613708, 0.49244638383388517, 0.4627079963684082, 0.4320942759513855, 0.40754836797714233, 0.38866173624992373, 0.3722610503435135, 0.35806763768196104, 0.34381386637687683, 0.32835571467876434, 0.313887482881546, 0.2978456974029541, 0.28028953075408936, 0.26205491423606875, 0.24302436411380768, 0.22394128143787384, 0.20554079115390778, 0.18561340272426602, 0.16209167838096616, 0.14397162199020386, 0.12474564164876933, 0.10150918066501613, 0.08697998523712158, 0.05724991485476494, 0.029584133997559547, 0.006172131001949355, -0.034059077501297};
                _cos_table0 = _mm512_loadu_ps(&cos_table[0]);
                _cos_table1 = _mm512_loadu_ps(&cos_table[16]);
                _cos_table2 = _mm512_loadu_ps(&cos_table[32]);
                _cos_table3 = _mm512_loadu_ps(&cos_table[48]);
/*                hnsw->graph_l0_finger.finger.neighboring_float_size = 16 * sizeof(float);
                hnsw->graph_l0_finger.finger.neighboring_index_size = 16 * sizeof(index_type);
                hnsw->graph_l0_finger.finger.center_size = 2 * sizeof(float);
                hnsw->graph_l0_finger.finger.num_dimension_blocks = hnsw->graph_l0_finger.finger.low_rank / 16;
*/
                std::string space_type = pecos::type_util::full_name<feat_vec_t>();
                if (space_type == "pecos::ann::FeatVecDenseL2Simd<float>") {
                    //approximate_distance = &approximate_l2_distance;
                    which = true;
                } 
                else if (space_type == "pecos::ann::FeatVecDenseIPSimd<float>") {
                    which = false;
                    //approximate_distance = &approximate_ip_distance;
                }
            }
            void compute_query_projection(float* query) {
                //uint32_t tmp;
                hnsw->graph_l0_finger.finger.compute_projection_information(query, query_projection.data(), query_norm, query_squared_norm);
                //hnsw->graph_l0_finger.finger.compute_query_rplsh_code(query_rplsh_code, query_projection.data());
                //_lookup_table = _mm512_set1_epi64(query_rplsh_code);
            }

            void approximate_distance(
                size_t neighbor_size, 
                const float& topk_ub_dist,
                const float& center_query_distance,
                const char* stored_info
            ) {
                if (which) {
                    approximate_l2_distance(
                        neighbor_size, 
                        topk_ub_dist,
                        center_query_distance,
                        stored_info
                    );
                } else {
                    approximate_angular_distance(
                        neighbor_size, 
                        topk_ub_dist,
                        center_query_distance,
                        stored_info
                    );
                } 

            }
            void approximate_angular_distance(
                size_t neighbor_size, 
                const float& topk_ub_dist,
                const float& center_query_ip_distance,
                const char* stored_info
            ) {
                // pass searcher to group_distance
                hnsw->graph_l0_finger.finger.approximate_angular_distance(
                    appx_dist.data(), 
                    hnsw->graph_l0_finger.max_degree,
                    topk_ub_dist,
                    neighbor_size, 
                    query_norm,
                    query_squared_norm,
                    query_projection.data(),
                    1.0 - center_query_ip_distance,
                    stored_info,
                    sss,
                    bbb
                );       
            }

            void approximate_l2_distance(
                size_t neighbor_size, 
                const float& topk_ub_dist,
                const float& center_query_l2_distance,
                const char* stored_info
            ) {
                // pass searcher to group_distance
                hnsw->graph_l0_finger.finger.approximate_distance(
                    appx_dist.data(), 
                    hnsw->graph_l0_finger.max_degree,
                    topk_ub_dist,
                    neighbor_size, 
                    query_norm,
                    query_squared_norm,
                    query_projection.data(),
                    center_query_l2_distance,
                    stored_info,
                    sss,
                    bbb
                );       
            }

            max_heap_t& search_level(const feat_vec_t& query, index_type init_node, index_type efS, index_type level) {
                if (scheme == "AppxOnly") {
                    return hnsw->search_level(query, init_node, efS, level, *this);
                } 
                else if ( scheme == "FINGER") {
                }
                else if (scheme == "MIX") {
                }
                else { 
                }
            }

            max_heap_t& predict_single(const feat_vec_t& query, index_type efS, index_type topk, index_type num_rerank) {
                return hnsw->predict_single(query, efS, topk, *this, num_rerank);
            }
        };

        Searcher create_searcher() const {
            return Searcher(this);
        }


        static nlohmann::json load_config(const std::string& filepath) {
            std::ifstream loadfile(filepath);
            std::string json_str;
            if (loadfile.is_open()) {
                json_str.assign(
                    std::istreambuf_iterator<char>(loadfile),
                    std::istreambuf_iterator<char>());
            } else {
                throw std::runtime_error("Unable to open config file at " + filepath);
            }
            auto j_param = nlohmann::json::parse(json_str);
            std::string hnsw_t_cur = pecos::type_util::full_name<HNSWFinger>();
            std::string hnsw_t_inp = j_param["hnsw_t"];
            if (hnsw_t_cur != hnsw_t_inp) {
                throw std::invalid_argument("Inconsistent HNSW_T: hnsw_t_cur = " + hnsw_t_cur  + " hnsw_t_cur = " + hnsw_t_inp);
            }
            return j_param;
        }

        void save_config(const std::string& filepath) const {
            nlohmann::json j_params = {
                {"hnsw_t", pecos::type_util::full_name<HNSWFinger>()},
                {"version", "v1.0"},
                {"train_params", {
                    {"num_node", this->num_node},
                    {"subspace_dimension", this->subspace_dimension},
                    {"sub_sample_points", this->sub_sample_points},
                    {"maxM", this->maxM},
                    {"maxM0", this->maxM0},
                    {"efC", this->efC},
                    {"max_level", this->max_level},
                    {"init_node", this->init_node}
                    }
                }
            };
            std::ofstream savefile(filepath, std::ofstream::trunc);
            if (savefile.is_open()) {
                savefile << j_params.dump(4);
                savefile.close();
            } else {
                throw std::runtime_error("Unable to save config file to " + filepath);
            }
        }

        void save(const std::string& model_dir) const {
            if (mkdir(model_dir.c_str(), 0777) == -1) {
                if (errno != EEXIST) {
                    throw std::runtime_error("Unable to create save folder at " + model_dir);
                }
            }
            save_config(model_dir + "/config.json");
            std::string index_path = model_dir + "/index.bin";
            FILE *fp = fopen(index_path.c_str(), "wb");
            pecos::file_util::fput_multiple<index_type>(&num_node, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&maxM, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&maxM0, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&efC, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&max_level, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&init_node, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&subspace_dimension, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&sub_sample_points, 1, fp);
            feature_vec.save(fp);
            graph_l1.save(fp);
            graph_l0_finger.save(fp);
            fclose(fp);
        }

        void load(const std::string& model_dir) {
            auto config = load_config(model_dir + "/config.json");
            std::string version = config.find("version") != config.end() ? config["version"] : "not found";
            std::string index_path = model_dir + "/index.bin";
            FILE *fp = fopen(index_path.c_str(), "rb");
            if (version == "v1.0") {
                pecos::file_util::fget_multiple<index_type>(&num_node, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&maxM, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&maxM0, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&efC, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&max_level, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&init_node, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&subspace_dimension, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&sub_sample_points, 1, fp);
                feature_vec.load(fp);
                graph_l1.load(fp);
                graph_l0_finger.load(fp);
            } else {
                throw std::runtime_error("Unable to load this binary with version = " + version);
            }
            fclose(fp);
        }

        template<class MAT_T>
        void train(
            const MAT_T &X_trn,
            index_type M,
            index_type efC,
            index_type subspace_dimension=0,
            index_type sub_sample_points=0,
            int threads=1,
            int max_level_upper_bound=-1
        ) {
            HNSW<dist_t, feat_vec_t>* hnsw = new HNSW<dist_t, feat_vec_t>();
            hnsw->train(X_trn, M, efC, threads, max_level_upper_bound);
            this->num_node = hnsw->num_node;
            this->maxM = hnsw->maxM;
            this->maxM0 = hnsw->maxM0;
            this->efC = hnsw->efC;
            this->max_level = hnsw->max_level;
            this->init_node = hnsw->init_node;
            this->subspace_dimension = subspace_dimension;
            this->sub_sample_points = sub_sample_points;

            graph_l1.num_node = hnsw->graph_l1.num_node;
            graph_l1.max_level = hnsw->graph_l1.max_level;
            graph_l1.max_degree = hnsw->graph_l1.max_degree;
            graph_l1.node_mem_size = hnsw->graph_l1.node_mem_size;
            graph_l1.level_mem_size = hnsw->graph_l1.level_mem_size;
            graph_l1.buffer.resize(hnsw->graph_l1.buffer.size());
            memcpy(graph_l1.buffer.data(), hnsw->graph_l1.buffer.data(), hnsw->graph_l1.buffer.size() * sizeof(index_type));

            //graph_l0_finger.build_quantizer(X_trn, subspace_dimension, sub_sample_points);
            graph_l0_finger.build_graph(hnsw->graph_l0);
            delete hnsw;
            feature_vec.init(X_trn, -1);
        }


        max_heap_t& predict_single(const feat_vec_t& query, index_type efS, index_type topk, Searcher& searcher, index_type num_rerank) const {
            index_type curr_node = this->init_node;
            auto &G1 = graph_l1;
            auto &G0 = feature_vec;
            // specialized search_level for level l=1,...,L because its faster for efS=1
            dist_t curr_dist = feat_vec_t::distance(
                query,
                G0.get_node_feat(init_node)
            );
            for (index_type curr_level = this->max_level; curr_level >= 1; curr_level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    const auto neighbors = G1.get_neighborhood(curr_node, curr_level);
                    if (neighbors.degree() != 0) {
                        feature_vec.prefetch_node_feat(neighbors[0]);
                        index_type max_j = neighbors.degree() - 1;
                        for (index_type j = 0; j <= max_j; j++) {
                            feature_vec.prefetch_node_feat(neighbors[std::min(j + 1, max_j)]);
                            auto next_node = neighbors[j];
                            dist_t next_dist = feat_vec_t::distance(
                                query,
                                G0.get_node_feat(next_node)
                            );
                            if (next_dist < curr_dist) {
                                curr_dist = next_dist;
                                curr_node = next_node;
                                changed = true;
                            }
                        }
                    }
                }
            }
            // generalized search_level for level=0 for efS >= 1
            searcher.search_level(query, curr_node, std::max(efS, topk), 0);
            auto& topk_queue = searcher.topk_queue;


            if (num_rerank > 0) {
                index_type t_size = topk_queue.size() > num_rerank ? topk_queue.size() - num_rerank : 0;
                for (index_type i = 0; i < t_size; i++) {
                    topk_queue.pop();
                }
                for (auto i = topk_queue.begin(); i != topk_queue.end(); ++i) {
                    feature_vec.prefetch_node_feat((*(i + 1)).node_id);
                    pair_t cand_pair = (*i);
                    dist_t next_dist = feat_vec_t::distance(
                        query,
                        G0.get_node_feat(cand_pair.node_id)
                    );
                    (*i).dist = next_dist;
                }
                std::sort(topk_queue.begin(), topk_queue.end());
                if (topk_queue.size() > topk) {
                    topk_queue.resize(topk);
                }
                return searcher.topk_queue;
            }



            if (topk < efS) {
                // remove extra when efS > topk
                while (topk_queue.size() > topk) {
                    topk_queue.pop();
                }
            }
            std::sort_heap(topk_queue.begin(), topk_queue.end());
            return topk_queue;
        }

        max_heap_t& search_level(
            const feat_vec_t& query,
            index_type init_node,
            index_type efS,
            index_type level,
            Searcher& searcher,
            std::vector<std::mutex>* mtx_nodes=nullptr
        ) const {
            const auto *G0_feature = &feature_vec;
            const auto *GFinger = &graph_l0_finger;
            searcher.reset();

            max_heap_t& topk_queue = searcher.topk_queue;
            min_heap_t& cand_queue = searcher.cand_queue;

            dist_t topk_ub_dist = feat_vec_t::distance(
                query,
                G0_feature->get_node_feat(init_node)
            );
            // compute query projection
            searcher.compute_query_projection(query.val); 

            topk_queue.emplace(topk_ub_dist, init_node);
            cand_queue.emplace(topk_ub_dist, init_node);
            searcher.mark_visited(init_node);
            // first stage, use the original exact distance to do inference.
            while (!cand_queue.empty()) {
                pair_t cand_pair = cand_queue.top();
                if (cand_pair.dist > topk_ub_dist) {
                    break;
                }
                cand_queue.pop();

                index_type cand_node = cand_pair.node_id;
                float center_query_l2_distance = cand_pair.dist; 
                // visiting neighbors of candidate node
                const auto neighbors = GFinger->get_neighborhood(cand_node, level);
                auto stored_info = GFinger->get_stored_info(cand_node);

                searcher.approximate_distance(
                    neighbors.degree(),
                    topk_ub_dist,
                    center_query_l2_distance,
                    stored_info
                );

                if (neighbors.degree() != 0) {
                    index_type max_j = neighbors.degree() - 1;


                    for (index_type j = 0; j <= max_j; j++) {
                        auto next_node = neighbors[j];
                        dist_t next_lb_dist = searcher.appx_dist[j];

                        if (topk_queue.size() < efS || next_lb_dist < topk_ub_dist) {
                            if (!searcher.is_visited(next_node)) {
                                searcher.mark_visited(next_node);
                                cand_queue.emplace(next_lb_dist, next_node);
                                topk_queue.emplace(next_lb_dist, next_node);
                                GFinger->prefetch_node_feat(cand_queue.top().node_id);
                                if (topk_queue.size() > efS) {
                                    topk_queue.pop();
                                }
                                if (!topk_queue.empty()) {
                                    topk_ub_dist = topk_queue.top().dist;
                                }
                            }
                        }
                    }

                    
                }

            }
            return topk_queue;
        }




        max_heap_t& search_mix(
            const feat_vec_t& query,
            index_type init_node,
            index_type efS,
            index_type level,
            Searcher& searcher,
            std::vector<std::mutex>* mtx_nodes=nullptr
        ) const {
            const auto *G0_feature = &feature_vec;
            const auto *GFinger = &graph_l0_finger;
            searcher.reset();

            max_heap_t& topk_queue = searcher.topk_queue;
            min_heap_t& cand_queue = searcher.cand_queue;

            dist_t topk_ub_dist = feat_vec_t::distance(
                query,
                G0_feature->get_node_feat(init_node)
            );
            // compute query projection
            searcher.compute_query_projection(query.val); 

            topk_queue.emplace(topk_ub_dist, init_node);
            cand_queue.emplace(topk_ub_dist, init_node);
            searcher.mark_visited(init_node);
            // first stage, use the original exact distance to do inference.
            size_t iteration_cnt = 0;
            while (!cand_queue.empty() ) {
                pair_t cand_pair = cand_queue.top();
                if (cand_pair.dist > topk_ub_dist) {
                    break;
                }
                cand_queue.pop();
                index_type cand_node = cand_pair.node_id;

                // visiting neighbors of candidate node
                const auto neighbors = GFinger->get_neighborhood(cand_node, level);
                if (neighbors.degree() != 0) {
                    //feature_vec.prefetch_node_feat(neighbors[0]);
                    index_type max_j = neighbors.degree() - 1;
                    for (index_type j = 0; j <= max_j; j++) {
                        feature_vec.prefetch_node_feat(neighbors[std::min(j + 1, max_j)]);
                        auto next_node = neighbors[j];
                        if (!searcher.is_visited(next_node)) {
                            searcher.mark_visited(next_node);
                            dist_t next_lb_dist;
                            next_lb_dist = feat_vec_t::distance(
                                query,
                                G0_feature->get_node_feat(next_node)
                            );
                            if (topk_queue.size() < efS || next_lb_dist < topk_ub_dist) {
                                cand_queue.emplace(next_lb_dist, next_node);
                                G0_feature->prefetch_node_feat(cand_queue.top().node_id);
                                topk_queue.emplace(next_lb_dist, next_node);
                                if (topk_queue.size() > efS) {
                                    topk_queue.pop();
                                }
                                if (!topk_queue.empty()) {
                                    topk_ub_dist = topk_queue.top().dist;
                                }
                            }
                        }
                    }
                    if (topk_queue.size() >= efS) {
                        break;
                    }
                }
                //iteration_cnt += 1;
            }
            // second stage, use approximate distance to scan 
           
            
            while (!cand_queue.empty()) {
                pair_t cand_pair = cand_queue.top();
                if (cand_pair.dist > topk_ub_dist) {
                    break;
                }
                cand_queue.pop();

                index_type cand_node = cand_pair.node_id;
                float center_query_l2_distance = cand_pair.dist; 
                // visiting neighbors of candidate node
                const auto neighbors = GFinger->get_neighborhood(cand_node, level);
                auto stored_info = GFinger->get_stored_info(cand_node);


                searcher.approximate_distance(
                    neighbors.degree(),
                    topk_ub_dist,
                    center_query_l2_distance,
                    stored_info
                );

                if (neighbors.degree() != 0) {
                    index_type max_j = neighbors.degree() - 1;


                    for (index_type j = 0; j <= max_j; j++) {
                        auto next_node = neighbors[j];
                        dist_t next_lb_dist = searcher.appx_dist[j];

                        if (topk_queue.size() < efS || next_lb_dist < topk_ub_dist) {
                            if (!searcher.is_visited(next_node)) {
                                searcher.mark_visited(next_node);
                                cand_queue.emplace(next_lb_dist, next_node);
                                topk_queue.emplace(next_lb_dist, next_node);
                                GFinger->prefetch_node_feat(cand_queue.top().node_id);
                                if (topk_queue.size() > efS) {
                                    topk_queue.pop();
                                }
                                if (!topk_queue.empty()) {
                                    topk_ub_dist = topk_queue.top().dist;
                                }
                            }
                        }
                    }

                }

            }
            return topk_queue;
        }














        max_heap_t& search_finger(
            const feat_vec_t& query,
            index_type init_node,
            index_type efS,
            index_type level,
            Searcher& searcher,
            std::vector<std::mutex>* mtx_nodes=nullptr
        ) const {
            const auto *G0_feature = &feature_vec;
            const auto *GFinger = &graph_l0_finger;
            searcher.reset();

            max_heap_t& topk_queue = searcher.topk_queue;
            min_heap_t& cand_queue = searcher.cand_queue;

            dist_t topk_ub_dist = feat_vec_t::distance(
                query,
                G0_feature->get_node_feat(init_node)
            );
            // compute query projection
            searcher.compute_query_projection(query.val); 

            topk_queue.emplace(topk_ub_dist, init_node);
            cand_queue.emplace(topk_ub_dist, init_node);
            searcher.mark_visited(init_node);
            // first stage, use the original exact distance to do inference.
            size_t iteration_cnt = 0;
            while (!cand_queue.empty() ) {
                pair_t cand_pair = cand_queue.top();
                if (cand_pair.dist > topk_ub_dist) {
                    break;
                }
                cand_queue.pop();
                index_type cand_node = cand_pair.node_id;

                // visiting neighbors of candidate node
                const auto neighbors = GFinger->get_neighborhood(cand_node, level);
                if (neighbors.degree() != 0) {
                    //feature_vec.prefetch_node_feat(neighbors[0]);
                    index_type max_j = neighbors.degree() - 1;
                    for (index_type j = 0; j <= max_j; j++) {
                        feature_vec.prefetch_node_feat(neighbors[std::min(j + 1, max_j)]);
                        auto next_node = neighbors[j];
                        if (!searcher.is_visited(next_node)) {
                            searcher.mark_visited(next_node);
                            dist_t next_lb_dist;
                            next_lb_dist = feat_vec_t::distance(
                                query,
                                G0_feature->get_node_feat(next_node)
                            );
                            if (topk_queue.size() < efS || next_lb_dist < topk_ub_dist) {
                                cand_queue.emplace(next_lb_dist, next_node);
                                G0_feature->prefetch_node_feat(cand_queue.top().node_id);
                                topk_queue.emplace(next_lb_dist, next_node);
                                if (topk_queue.size() > efS) {
                                    topk_queue.pop();
                                }
                                if (!topk_queue.empty()) {
                                    topk_ub_dist = topk_queue.top().dist;
                                }
                            }
                        }
                    }
                    if (topk_queue.size() >= efS) {
                        break;
                    }
                }
                //iteration_cnt += 1;
            }
            // second stage, use approximate distance to scan 
           
            
            while (!cand_queue.empty()) {
                pair_t cand_pair = cand_queue.top();
                if (cand_pair.dist > topk_ub_dist) {
                    break;
                }
                cand_queue.pop();

                index_type cand_node = cand_pair.node_id;
                float center_query_l2_distance = cand_pair.dist; 
                // visiting neighbors of candidate node
                const auto neighbors = GFinger->get_neighborhood(cand_node, level);
                auto stored_info = GFinger->get_stored_info(cand_node);


                searcher.approximate_distance(
                    neighbors.degree(),
                    topk_ub_dist,
                    center_query_l2_distance,
                    stored_info
                );


                if (neighbors.degree() != 0) {
                    index_type max_j = neighbors.degree() - 1;

                    for (index_type j = 0; j <= max_j; j++) {
                        auto next_node = neighbors[j];
                        if (searcher.appx_dist[j] and !searcher.is_visited(next_node)) {
                            G0_feature->prefetch_node_feat(next_node);
                            searcher.mark_visited(next_node);
                        } else { 
                            searcher.appx_dist[j] = 0;
                        }
                        
                    }

                    //feature_vec.prefetch_node_feat(neighbors[0]);
                    
                    for (index_type j = 0; j <= max_j; j++) {
                        auto next_node = neighbors[j];
                        dist_t next_lb_dist;
                        next_lb_dist = searcher.appx_dist[j];
                          if (next_lb_dist) {
                                next_lb_dist = feat_vec_t::distance(
                                    query,
                                    G0_feature->get_node_feat(next_node)
                                );
                                cand_queue.emplace(next_lb_dist, next_node);
                                topk_queue.emplace(next_lb_dist, next_node);
                            }
                    }
                    //G0_feature->prefetch_node_feat(cand_queue.top().node_id);
                    GFinger->prefetch_node_feat(cand_queue.top().node_id);
                    while (topk_queue.size() > efS) {
                        topk_queue.pop();
                    }
                    topk_ub_dist = topk_queue.top().dist;
                                
                    
                }

            }
            return topk_queue;
        }
    };
