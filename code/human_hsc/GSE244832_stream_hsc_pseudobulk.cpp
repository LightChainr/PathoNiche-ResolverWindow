#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr int kDonors = 18;
constexpr int kHscCluster = 3;

class FastInput {
 public:
  bool next_int(int& value) {
    int c;
    do {
      c = read_char();
      if (c == EOF) return false;
    } while (c < '0' || c > '9');
    value = 0;
    do {
      value = value * 10 + (c - '0');
      c = read_char();
    } while (c >= '0' && c <= '9');
    return true;
  }

 private:
  static constexpr size_t kBufferSize = 1 << 20;
  std::array<char, kBufferSize> buffer_{};
  size_t position_ = 0;
  size_t size_ = 0;

  int read_char() {
    if (position_ == size_) {
      size_ = std::fread(buffer_.data(), 1, buffer_.size(), stdin);
      position_ = 0;
      if (size_ == 0) return EOF;
    }
    return buffer_[position_++];
  }
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: stream_hsc cell_map.tsv genes.tsv counts.tsv metadata.tsv\n";
    return 2;
  }
  std::ifstream map_file(argv[1]);
  std::vector<int> donor_by_cell;
  std::vector<int> cluster_by_cell;
  int donor = 0;
  int cluster = 0;
  while (map_file >> donor >> cluster) {
    donor_by_cell.push_back(donor);
    cluster_by_cell.push_back(cluster);
  }

  std::ifstream gene_file(argv[2]);
  std::vector<std::string> genes;
  std::string gene;
  while (std::getline(gene_file, gene)) {
    if (!gene.empty()) genes.push_back(gene);
  }
  const size_t n_genes = genes.size();
  std::vector<std::uint64_t> counts(n_genes * kDonors, 0);

  FastInput input;
  int gene_index = 0;
  int cell_index = 0;
  int count = 0;
  std::uint64_t entries = 0;
  while (input.next_int(gene_index) && input.next_int(cell_index) && input.next_int(count)) {
    --gene_index;
    --cell_index;
    if (gene_index < 0 || static_cast<size_t>(gene_index) >= n_genes || cell_index < 0 ||
        static_cast<size_t>(cell_index) >= donor_by_cell.size()) {
      std::cerr << "matrix index out of range at entry " << entries << "\n";
      return 3;
    }
    if (cluster_by_cell[cell_index] == kHscCluster) {
      counts[static_cast<size_t>(gene_index) * kDonors + donor_by_cell[cell_index]] += count;
    }
    ++entries;
  }

  std::ofstream output(argv[3]);
  output << "gene";
  for (int index = 0; index < kDonors; ++index) output << "\tdonor_" << index;
  output << '\n';
  for (size_t gene_id = 0; gene_id < n_genes; ++gene_id) {
    output << genes[gene_id];
    for (int index = 0; index < kDonors; ++index) {
      output << '\t' << counts[gene_id * kDonors + index];
    }
    output << '\n';
  }

  std::ofstream audit(argv[4]);
  audit << "matrix_entries\t" << entries << "\n";
  audit << "matrix_cells\t" << donor_by_cell.size() << "\n";
  audit << "matrix_genes\t" << n_genes << "\n";
  return 0;
}
